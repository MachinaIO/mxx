import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard477

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70212
def owner : Owner := ⟨.program ⟨214⟩, ⟨19527⟩⟩
def transferEvent : Nat := 70212
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 70211) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70211)
      LeftBound70211.bound (LeftBound70211.actual selector witness) := by
  exact .transfer (LeftBound70211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound70211.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound70211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound70211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70212

namespace LeftBound70291
def owner : Owner := ⟨.program ⟨214⟩, ⟨14199⟩⟩
def transferEvent : Nat := 70291
def frameStart : Nat := 70262
def rule : BoundRule := .product (.predecessor 0 70289 .coefficient) (.predecessor 1 70290 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70289 .coefficient)
      LeftAuthority70287.bound (LeftAuthority70287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70287.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70290 .coefficient)
      LeftAuthority70284.bound (LeftAuthority70284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70287.bound LeftAuthority70284.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70287.bound, LeftAuthority70284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70287.actual selector witness) * (LeftAuthority70284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70291

namespace LeftBound70295
def owner : Owner := ⟨.program ⟨214⟩, ⟨14200⟩⟩
def transferEvent : Nat := 70295
def frameStart : Nat := 70262
def rule : BoundRule := .identity (.predecessor 0 70294 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70294 .coefficient)
      LeftBound70291.bound (LeftBound70291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70291.derived selector witness)

def rawBound : CoeffClass := LeftBound70291.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70291.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70295

namespace LeftBound70312
def owner : Owner := ⟨.program ⟨214⟩, ⟨14310⟩⟩
def transferEvent : Nat := 70312
def frameStart : Nat := 70262
def rule : BoundRule := .sum [.predecessor 0 70310 .coefficient, .predecessor 1 70311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70310 .coefficient)
      LeftBound70295.bound (LeftBound70295.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70311 .coefficient)
      LeftAuthority70308.bound (LeftAuthority70308.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70308.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70295.bound, LeftAuthority70308.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70295.bound, LeftAuthority70308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70295.actual selector witness, LeftAuthority70308.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70312

namespace LeftBound70315
def owner : Owner := ⟨.program ⟨214⟩, ⟨14311⟩⟩
def transferEvent : Nat := 70315
def frameStart : Nat := 70262
def rule : BoundRule := .identity (.predecessor 0 70314 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70314 .coefficient)
      LeftBound70312.bound (LeftBound70312.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70312.derived selector witness)

def rawBound : CoeffClass := LeftBound70312.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70312.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70315

namespace LeftBound70321
def owner : Owner := ⟨.program ⟨214⟩, ⟨14312⟩⟩
def transferEvent : Nat := 70321
def frameStart : Nat := 70262
def rule : BoundRule := .product (.predecessor 0 70319 .coefficient) (.predecessor 1 70320 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70319 .coefficient)
      LeftAuthority70317.bound (LeftAuthority70317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70320 .coefficient)
      LeftBound70315.bound (LeftBound70315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority70317.bound LeftBound70315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70317.bound, LeftBound70315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority70317.actual selector witness) * (LeftBound70315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70321

namespace LeftBound70337
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 70337
def frameStart : Nat := 70262
def rule : BoundRule := .scale (.predecessor 0 70335 .coefficient) (.value (.predecessor 1 70336 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70335 .coefficient)
      LeftAuthority70333.bound (LeftAuthority70333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70336 .coefficient)
      LeftAuthority70324.bound (LeftAuthority70324.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70324.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70333.bound LeftAuthority70324.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70333.bound, LeftAuthority70324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70333.actual selector witness) * (LeftAuthority70324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70337

namespace LeftBound70340
def owner : Owner := ⟨.program ⟨214⟩, ⟨6759⟩⟩
def transferEvent : Nat := 70340
def frameStart : Nat := 70262
def rule : BoundRule := .identity (.predecessor 0 70339 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70339 .coefficient)
      LeftAuthority70327.bound (LeftAuthority70327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70327.derived selector witness)

def rawBound : CoeffClass := LeftAuthority70327.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority70327.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70340

namespace LeftBound70344
def owner : Owner := ⟨.program ⟨214⟩, ⟨7854⟩⟩
def transferEvent : Nat := 70344
def frameStart : Nat := 70262
def rule : BoundRule := .product (.predecessor 0 70342 .coefficient) (.predecessor 1 70343 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70342 .coefficient)
      LeftBound70340.bound (LeftBound70340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70343 .coefficient)
      LeftBound70337.bound (LeftBound70337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70337.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70340.bound LeftBound70337.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70340.bound, LeftBound70337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70340.actual selector witness) * (LeftBound70337.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70344

namespace LeftBound70349
def owner : Owner := ⟨.program ⟨214⟩, ⟨14313⟩⟩
def transferEvent : Nat := 70349
def frameStart : Nat := 70262
def rule : BoundRule := .sum [.predecessor 0 70347 .coefficient, .predecessor 1 70348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70347 .coefficient)
      LeftBound70344.bound (LeftBound70344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70348 .coefficient)
      LeftBound70321.bound (LeftBound70321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70344.bound, LeftBound70321.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70344.bound, LeftBound70321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70344.actual selector witness, LeftBound70321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70349

namespace LeftBound70353
def owner : Owner := ⟨.program ⟨214⟩, ⟨26064⟩⟩
def transferEvent : Nat := 70353
def frameStart : Nat := 70262
def rule : BoundRule := .product (.predecessor 0 70351 .coefficient) (.predecessor 1 70352 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70351 .coefficient)
      LeftBound70349.bound (LeftBound70349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70352 .coefficient)
      LeftAuthority70306.bound (LeftAuthority70306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70306.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70349.bound LeftAuthority70306.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70349.bound, LeftAuthority70306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70349.actual selector witness) * (LeftAuthority70306.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70353

namespace LeftBound70364
def owner : Owner := ⟨.program ⟨214⟩, ⟨15938⟩⟩
def transferEvent : Nat := 70364
def frameStart : Nat := 70262
def rule : BoundRule := .product (.predecessor 0 70362 .coefficient) (.predecessor 1 70363 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70362 .coefficient)
      LeftAuthority70317.bound (LeftAuthority70317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70317.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70363 .coefficient)
      LeftAuthority70360.bound (LeftAuthority70360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70317.bound LeftAuthority70360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70317.bound, LeftAuthority70360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70317.actual selector witness) * (LeftAuthority70360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70364

namespace LeftBound70372
def owner : Owner := ⟨.program ⟨214⟩, ⟨15939⟩⟩
def transferEvent : Nat := 70372
def frameStart : Nat := 70262
def rule : BoundRule := .sum [.predecessor 0 70370 .coefficient, .predecessor 1 70371 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70370 .coefficient)
      LeftAuthority70368.bound (LeftAuthority70368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70371 .coefficient)
      LeftBound70364.bound (LeftBound70364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70368.bound, LeftBound70364.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70368.bound, LeftBound70364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70368.actual selector witness, LeftBound70364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70372

namespace LeftBound70376
def owner : Owner := ⟨.program ⟨214⟩, ⟨26065⟩⟩
def transferEvent : Nat := 70376
def frameStart : Nat := 70262
def rule : BoundRule := .sum [.predecessor 0 70374 .coefficient, .predecessor 1 70375 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70374 .coefficient)
      LeftBound70372.bound (LeftBound70372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70372.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70375 .coefficient)
      LeftBound70353.bound (LeftBound70353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70372.bound, LeftBound70353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70372.bound, LeftBound70353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70372.actual selector witness, LeftBound70353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70376

namespace LeftBound70389
def owner : Owner := ⟨.program ⟨214⟩, ⟨26063⟩⟩
def transferEvent : Nat := 70389
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70387 .coefficient, .predecessor 1 70388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70387 .coefficient)
      LeftBound70210.bound (LeftBound70210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70388 .coefficient)
      LeftBound70193.bound (LeftBound70193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70210.bound, LeftBound70193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70210.bound, LeftBound70193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70210.actual selector witness, LeftBound70193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70389

namespace LeftBound70392
def owner : Owner := ⟨.program ⟨214⟩, ⟨26063⟩⟩
def transferEvent : Nat := 70392
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70386 .summary, .result 70200 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70386 .summary)
      LeftBound70212.bound (LeftBound70212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19527⟩⟩) (rawTerms := some (Proof.Events274.exact70386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70200 .summary)
      LeftBound70195.bound (LeftBound70195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26062⟩⟩) (rawTerms := some (Proof.Events274.exact70200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70195.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70212.bound, LeftBound70195.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70212.bound, LeftBound70195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70212.actual selector witness, LeftBound70195.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70392

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
