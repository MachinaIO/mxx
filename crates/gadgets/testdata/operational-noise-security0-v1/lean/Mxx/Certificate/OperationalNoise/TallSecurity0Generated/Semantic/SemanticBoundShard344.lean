import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard343

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound51328
def owner : Owner := ⟨.program ⟨214⟩, ⟨13163⟩⟩
def transferEvent : Nat := 51328
def frameStart : Nat := 51299
def rule : BoundRule := .product (.predecessor 0 51326 .coefficient) (.predecessor 1 51327 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51326 .coefficient)
      LeftAuthority51324.bound (LeftAuthority51324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51327 .coefficient)
      LeftAuthority51321.bound (LeftAuthority51321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51321.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51321.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51324.bound LeftAuthority51321.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51324.bound, LeftAuthority51321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51324.actual selector witness) * (LeftAuthority51321.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51328

namespace LeftBound51332
def owner : Owner := ⟨.program ⟨214⟩, ⟨13164⟩⟩
def transferEvent : Nat := 51332
def frameStart : Nat := 51299
def rule : BoundRule := .identity (.predecessor 0 51331 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51331 .coefficient)
      LeftBound51328.bound (LeftBound51328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51328.derived selector witness)

def rawBound : CoeffClass := LeftBound51328.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51328.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51332

namespace LeftBound51349
def owner : Owner := ⟨.program ⟨214⟩, ⟨13254⟩⟩
def transferEvent : Nat := 51349
def frameStart : Nat := 51299
def rule : BoundRule := .sum [.predecessor 0 51347 .coefficient, .predecessor 1 51348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51347 .coefficient)
      LeftBound51332.bound (LeftBound51332.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51348 .coefficient)
      LeftAuthority51345.bound (LeftAuthority51345.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51332.bound, LeftAuthority51345.bound]
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51332.bound, LeftAuthority51345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51332.actual selector witness, LeftAuthority51345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51349

namespace LeftBound51352
def owner : Owner := ⟨.program ⟨214⟩, ⟨13255⟩⟩
def transferEvent : Nat := 51352
def frameStart : Nat := 51299
def rule : BoundRule := .identity (.predecessor 0 51351 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51351 .coefficient)
      LeftBound51349.bound (LeftBound51349.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51349.derived selector witness)

def rawBound : CoeffClass := LeftBound51349.bound
def bound : CoeffClass := .finite ⟨3364, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51349.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51352

namespace LeftBound51358
def owner : Owner := ⟨.program ⟨214⟩, ⟨13256⟩⟩
def transferEvent : Nat := 51358
def frameStart : Nat := 51299
def rule : BoundRule := .product (.predecessor 0 51356 .coefficient) (.predecessor 1 51357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51356 .coefficient)
      LeftAuthority51354.bound (LeftAuthority51354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51357 .coefficient)
      LeftBound51352.bound (LeftBound51352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51352.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority51354.bound LeftBound51352.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51354.bound, LeftBound51352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority51354.actual selector witness) * (LeftBound51352.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51358

namespace LeftBound51374
def owner : Owner := ⟨.program ⟨214⟩, ⟨7880⟩⟩
def transferEvent : Nat := 51374
def frameStart : Nat := 51299
def rule : BoundRule := .scale (.predecessor 0 51372 .coefficient) (.value (.predecessor 1 51373 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51372 .coefficient)
      LeftAuthority51370.bound (LeftAuthority51370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51373 .coefficient)
      LeftAuthority51361.bound (LeftAuthority51361.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51361.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority51370.bound LeftAuthority51361.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51370.bound, LeftAuthority51361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority51370.actual selector witness) * (LeftAuthority51361.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound51374

namespace LeftBound51377
def owner : Owner := ⟨.program ⟨214⟩, ⟨6769⟩⟩
def transferEvent : Nat := 51377
def frameStart : Nat := 51299
def rule : BoundRule := .identity (.predecessor 0 51376 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51376 .coefficient)
      LeftAuthority51364.bound (LeftAuthority51364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51364.derived selector witness)

def rawBound : CoeffClass := LeftAuthority51364.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority51364.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51377

namespace LeftBound51381
def owner : Owner := ⟨.program ⟨214⟩, ⟨7881⟩⟩
def transferEvent : Nat := 51381
def frameStart : Nat := 51299
def rule : BoundRule := .product (.predecessor 0 51379 .coefficient) (.predecessor 1 51380 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51379 .coefficient)
      LeftBound51377.bound (LeftBound51377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51380 .coefficient)
      LeftBound51374.bound (LeftBound51374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51374.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51377.bound LeftBound51374.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51377.bound, LeftBound51374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51377.actual selector witness) * (LeftBound51374.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51381

namespace LeftBound51386
def owner : Owner := ⟨.program ⟨214⟩, ⟨13257⟩⟩
def transferEvent : Nat := 51386
def frameStart : Nat := 51299
def rule : BoundRule := .sum [.predecessor 0 51384 .coefficient, .predecessor 1 51385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51384 .coefficient)
      LeftBound51381.bound (LeftBound51381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51385 .coefficient)
      LeftBound51358.bound (LeftBound51358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51358.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51381.bound, LeftBound51358.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51381.bound, LeftBound51358.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51381.actual selector witness, LeftBound51358.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51386

namespace LeftBound51390
def owner : Owner := ⟨.program ⟨214⟩, ⟨25689⟩⟩
def transferEvent : Nat := 51390
def frameStart : Nat := 51299
def rule : BoundRule := .product (.predecessor 0 51388 .coefficient) (.predecessor 1 51389 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51388 .coefficient)
      LeftBound51386.bound (LeftBound51386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51389 .coefficient)
      LeftAuthority51343.bound (LeftAuthority51343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51386.bound LeftAuthority51343.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51386.bound, LeftAuthority51343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51386.actual selector witness) * (LeftAuthority51343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51390

namespace LeftBound51401
def owner : Owner := ⟨.program ⟨214⟩, ⟨16877⟩⟩
def transferEvent : Nat := 51401
def frameStart : Nat := 51299
def rule : BoundRule := .product (.predecessor 0 51399 .coefficient) (.predecessor 1 51400 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51399 .coefficient)
      LeftAuthority51354.bound (LeftAuthority51354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51354.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51400 .coefficient)
      LeftAuthority51397.bound (LeftAuthority51397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority51354.bound LeftAuthority51397.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51354.bound, LeftAuthority51397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority51354.actual selector witness) * (LeftAuthority51397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51401

namespace LeftBound51409
def owner : Owner := ⟨.program ⟨214⟩, ⟨16878⟩⟩
def transferEvent : Nat := 51409
def frameStart : Nat := 51299
def rule : BoundRule := .sum [.predecessor 0 51407 .coefficient, .predecessor 1 51408 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51407 .coefficient)
      LeftAuthority51405.bound (LeftAuthority51405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51408 .coefficient)
      LeftBound51401.bound (LeftBound51401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority51405.bound, LeftBound51401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51405.bound, LeftBound51401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority51405.actual selector witness, LeftBound51401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51409

namespace LeftBound51413
def owner : Owner := ⟨.program ⟨214⟩, ⟨25690⟩⟩
def transferEvent : Nat := 51413
def frameStart : Nat := 51299
def rule : BoundRule := .sum [.predecessor 0 51411 .coefficient, .predecessor 1 51412 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51411 .coefficient)
      LeftBound51409.bound (LeftBound51409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51412 .coefficient)
      LeftBound51390.bound (LeftBound51390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51409.bound, LeftBound51390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51409.bound, LeftBound51390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51409.actual selector witness, LeftBound51390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51413

namespace LeftBound51426
def owner : Owner := ⟨.program ⟨214⟩, ⟨25688⟩⟩
def transferEvent : Nat := 51426
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 51424 .coefficient, .predecessor 1 51425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51424 .coefficient)
      LeftBound51247.bound (LeftBound51247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51425 .coefficient)
      LeftBound51230.bound (LeftBound51230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51247.bound, LeftBound51230.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51247.bound, LeftBound51230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51247.actual selector witness, LeftBound51230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51426

namespace LeftBound51429
def owner : Owner := ⟨.program ⟨214⟩, ⟨25688⟩⟩
def transferEvent : Nat := 51429
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 51423 .summary, .result 51237 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51423 .summary)
      LeftBound51249.bound (LeftBound51249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20183⟩⟩) (rawTerms := some (Proof.Events200.exact51423RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 51237 .summary)
      LeftBound51232.bound (LeftBound51232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25687⟩⟩) (rawTerms := some (Proof.Events200.exact51237RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51232.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51249.bound, LeftBound51232.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51249.bound, LeftBound51232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51249.actual selector witness, LeftBound51232.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51429

namespace LeftBound51433
def owner : Owner := ⟨.program ⟨214⟩, ⟨29834⟩⟩
def transferEvent : Nat := 51433
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 51431 .coefficient) (.predecessor 1 51432 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51431 .coefficient)
      LeftBound51426.bound (LeftBound51426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events200.exact51430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51432 .coefficient)
      LeftAuthority51152.bound (LeftAuthority51152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51152.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound51426.bound LeftAuthority51152.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51426.bound, LeftAuthority51152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound51426.actual selector witness) * (LeftAuthority51152.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51433

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
