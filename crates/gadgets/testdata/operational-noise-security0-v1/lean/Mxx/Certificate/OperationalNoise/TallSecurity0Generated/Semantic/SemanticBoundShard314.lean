import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard257
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard313

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound47334
def owner : Owner := ⟨.program ⟨214⟩, ⟨16598⟩⟩
def transferEvent : Nat := 47334
def frameStart : Nat := 47275
def rule : BoundRule := .identity (.predecessor 0 47333 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47333 .coefficient)
      LeftBound47331.bound (LeftBound47331.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound47331.derived selector witness)

def rawBound : CoeffClass := LeftBound47331.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound47331.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound47334

namespace LeftBound47340
def owner : Owner := ⟨.program ⟨214⟩, ⟨16599⟩⟩
def transferEvent : Nat := 47340
def frameStart : Nat := 47275
def rule : BoundRule := .product (.predecessor 0 47338 .coefficient) (.predecessor 1 47339 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47338 .coefficient)
      LeftAuthority47336.bound (LeftAuthority47336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47339 .coefficient)
      LeftBound47334.bound (LeftBound47334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47334.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority47336.bound LeftBound47334.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47336.bound, LeftBound47334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority47336.actual selector witness) * (LeftBound47334.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47340

namespace LeftBound47348
def owner : Owner := ⟨.program ⟨214⟩, ⟨16600⟩⟩
def transferEvent : Nat := 47348
def frameStart : Nat := 47275
def rule : BoundRule := .sum [.predecessor 0 47346 .coefficient, .predecessor 1 47347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47346 .coefficient)
      LeftAuthority47344.bound (LeftAuthority47344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47347 .coefficient)
      LeftBound47340.bound (LeftBound47340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47344.bound, LeftBound47340.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47344.bound, LeftBound47340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47344.actual selector witness, LeftBound47340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47348

namespace LeftBound47352
def owner : Owner := ⟨.program ⟨214⟩, ⟨29188⟩⟩
def transferEvent : Nat := 47352
def frameStart : Nat := 47275
def rule : BoundRule := .product (.predecessor 0 47350 .coefficient) (.predecessor 1 47351 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47350 .coefficient)
      LeftBound47348.bound (LeftBound47348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47351 .coefficient)
      LeftAuthority47325.bound (LeftAuthority47325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47325.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47348.bound LeftAuthority47325.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47348.bound, LeftAuthority47325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47348.actual selector witness) * (LeftAuthority47325.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47352

namespace LeftBound47363
def owner : Owner := ⟨.program ⟨214⟩, ⟨17959⟩⟩
def transferEvent : Nat := 47363
def frameStart : Nat := 47275
def rule : BoundRule := .product (.predecessor 0 47361 .coefficient) (.predecessor 1 47362 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47361 .coefficient)
      LeftAuthority47336.bound (LeftAuthority47336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47362 .coefficient)
      LeftAuthority47359.bound (LeftAuthority47359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority47336.bound LeftAuthority47359.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47336.bound, LeftAuthority47359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority47336.actual selector witness) * (LeftAuthority47359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47363

namespace LeftBound47371
def owner : Owner := ⟨.program ⟨214⟩, ⟨17960⟩⟩
def transferEvent : Nat := 47371
def frameStart : Nat := 47275
def rule : BoundRule := .sum [.predecessor 0 47369 .coefficient, .predecessor 1 47370 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47369 .coefficient)
      LeftAuthority47367.bound (LeftAuthority47367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47370 .coefficient)
      LeftBound47363.bound (LeftBound47363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority47367.bound, LeftBound47363.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47367.bound, LeftBound47363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority47367.actual selector witness, LeftBound47363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47371

namespace LeftBound47375
def owner : Owner := ⟨.program ⟨214⟩, ⟨29193⟩⟩
def transferEvent : Nat := 47375
def frameStart : Nat := 47275
def rule : BoundRule := .sum [.predecessor 0 47373 .coefficient, .predecessor 1 47374 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47373 .coefficient)
      LeftBound47371.bound (LeftBound47371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47371.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47374 .coefficient)
      LeftBound47352.bound (LeftBound47352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47371.bound, LeftBound47352.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47371.bound, LeftBound47352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47371.actual selector witness, LeftBound47352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47375

namespace LeftBound47388
def owner : Owner := ⟨.program ⟨214⟩, ⟨29190⟩⟩
def transferEvent : Nat := 47388
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 47386 .coefficient, .predecessor 1 47387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47386 .coefficient)
      LeftBound47217.bound (LeftBound47217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47217.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47387 .coefficient)
      LeftBound47200.bound (LeftBound47200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events184.exact47207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47217.bound, LeftBound47200.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47217.bound, LeftBound47200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47217.actual selector witness, LeftBound47200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47388

namespace LeftBound47391
def owner : Owner := ⟨.program ⟨214⟩, ⟨29190⟩⟩
def transferEvent : Nat := 47391
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 47385 .summary, .result 47207 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47385 .summary)
      LeftBound47219.bound (LeftBound47219.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22203⟩⟩) (rawTerms := some (Proof.Events185.exact47385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47207 .summary)
      LeftBound47202.bound (LeftBound47202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29189⟩⟩) (rawTerms := some (Proof.Events184.exact47207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound47219.bound, LeftBound47202.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47219.bound, LeftBound47202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound47219.actual selector witness, LeftBound47202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound47391

namespace LeftBound47395
def owner : Owner := ⟨.program ⟨214⟩, ⟨29191⟩⟩
def transferEvent : Nat := 47395
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47393 .coefficient) (.predecessor 1 47394 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47393 .coefficient)
      LeftBound47388.bound (LeftBound47388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound47388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound47388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47394 .coefficient)
      LeftBound5598.bound (LeftBound5598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5598.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47388.bound LeftBound5598.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47388.bound, LeftBound5598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47388.actual selector witness) * (LeftBound5598.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47395

namespace LeftBound47396
def owner : Owner := ⟨.program ⟨214⟩, ⟨29191⟩⟩
def transferEvent : Nat := 47396
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6667⟩⟩]⟩ [⟨.result 5595 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5595 .coefficient)
      LeftAuthority5594.bound (LeftAuthority5594.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6667⟩⟩) (rawTerms := some (Proof.Events021.exact5595RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5594.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5594.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5594.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5594.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47396

namespace LeftBound47397
def owner : Owner := ⟨.program ⟨214⟩, ⟨29191⟩⟩
def transferEvent : Nat := 47397
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 47392 .summary) (.transfer 47396) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47392 .summary)
      LeftBound47391.bound (LeftBound47391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29190⟩⟩) (rawTerms := some (Proof.Events185.exact47392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound47391.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47396)
      LeftBound47396.bound (LeftBound47396.actual selector witness) := by
  exact .transfer (LeftBound47396.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound47391.bound LeftBound47396.bound
def bound : CoeffClass := .finite ⟨4742899020835760917459238912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound47391.bound, LeftBound47396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound47391.actual selector witness) * (LeftBound47396.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47397

namespace LeftBound47412
def owner : Owner := ⟨.program ⟨214⟩, ⟨28972⟩⟩
def transferEvent : Nat := 47412
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 47410 .coefficient) (.predecessor 1 47411 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47410 .coefficient)
      LeftBound38729.bound (LeftBound38729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events151.exact38733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47411 .coefficient)
      LeftAuthority47408.bound (LeftAuthority47408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38729.bound LeftAuthority47408.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38729.bound, LeftAuthority47408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38729.actual selector witness) * (LeftAuthority47408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47412

namespace LeftBound47413
def owner : Owner := ⟨.program ⟨214⟩, ⟨28972⟩⟩
def transferEvent : Nat := 47413
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28970⟩⟩]⟩ [⟨.result 47409 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 47409 .coefficient)
      LeftAuthority47408.bound (LeftAuthority47408.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28970⟩⟩) (rawTerms := some (Proof.Events185.exact47409RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47408.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority47408.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47408.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound47413

namespace LeftBound47414
def owner : Owner := ⟨.program ⟨214⟩, ⟨28972⟩⟩
def transferEvent : Nat := 47414
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38733 .summary) (.transfer 47413) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38733 .summary)
      LeftBound38732.bound (LeftBound38732.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25385⟩⟩) (rawTerms := some (Proof.Events151.exact38733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 47413)
      LeftBound47413.bound (LeftBound47413.actual selector witness) := by
  exact .transfer (LeftBound47413.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38732.bound LeftBound47413.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38732.bound, LeftBound47413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38732.actual selector witness) * (LeftBound47413.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound47414

namespace LeftBound47425
def owner : Owner := ⟨.program ⟨214⟩, ⟨22058⟩⟩
def transferEvent : Nat := 47425
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 47423 .coefficient) (.value (.predecessor 1 47424 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 47423 .coefficient)
      LeftAuthority47421.bound (LeftAuthority47421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events185.exact47422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority47421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority47421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 47424 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority47421.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority47421.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority47421.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound47425

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
