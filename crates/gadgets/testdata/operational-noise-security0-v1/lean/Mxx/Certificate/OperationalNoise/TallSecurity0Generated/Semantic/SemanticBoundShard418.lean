import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard366
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard417

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62380
def owner : Owner := ⟨.program ⟨214⟩, ⟨16425⟩⟩
def transferEvent : Nat := 62380
def frameStart : Nat := 62324
def rule : BoundRule := .sum [.predecessor 0 62378 .coefficient, .predecessor 1 62379 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62378 .coefficient)
      LeftBound62363.bound (LeftBound62363.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62379 .coefficient)
      LeftAuthority62376.bound (LeftAuthority62376.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority62376.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62363.bound, LeftAuthority62376.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62363.bound, LeftAuthority62376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62363.actual selector witness, LeftAuthority62376.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62380

namespace LeftBound62383
def owner : Owner := ⟨.program ⟨214⟩, ⟨16426⟩⟩
def transferEvent : Nat := 62383
def frameStart : Nat := 62324
def rule : BoundRule := .identity (.predecessor 0 62382 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62382 .coefficient)
      LeftBound62380.bound (LeftBound62380.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62380.derived selector witness)

def rawBound : CoeffClass := LeftBound62380.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound62380.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62383

namespace LeftBound62389
def owner : Owner := ⟨.program ⟨214⟩, ⟨16427⟩⟩
def transferEvent : Nat := 62389
def frameStart : Nat := 62324
def rule : BoundRule := .product (.predecessor 0 62387 .coefficient) (.predecessor 1 62388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62387 .coefficient)
      LeftAuthority62385.bound (LeftAuthority62385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62385.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62388 .coefficient)
      LeftBound62383.bound (LeftBound62383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62383.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority62385.bound LeftBound62383.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62385.bound, LeftBound62383.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority62385.actual selector witness) * (LeftBound62383.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62389

namespace LeftBound62397
def owner : Owner := ⟨.program ⟨214⟩, ⟨16428⟩⟩
def transferEvent : Nat := 62397
def frameStart : Nat := 62324
def rule : BoundRule := .sum [.predecessor 0 62395 .coefficient, .predecessor 1 62396 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62395 .coefficient)
      LeftAuthority62393.bound (LeftAuthority62393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62393.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62396 .coefficient)
      LeftBound62389.bound (LeftBound62389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62393.bound, LeftBound62389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62393.bound, LeftBound62389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62393.actual selector witness, LeftBound62389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62397

namespace LeftBound62401
def owner : Owner := ⟨.program ⟨214⟩, ⟨28741⟩⟩
def transferEvent : Nat := 62401
def frameStart : Nat := 62324
def rule : BoundRule := .product (.predecessor 0 62399 .coefficient) (.predecessor 1 62400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62399 .coefficient)
      LeftBound62397.bound (LeftBound62397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62400 .coefficient)
      LeftAuthority62374.bound (LeftAuthority62374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62374.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62374.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62397.bound LeftAuthority62374.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62397.bound, LeftAuthority62374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62397.actual selector witness) * (LeftAuthority62374.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62401

namespace LeftBound62412
def owner : Owner := ⟨.program ⟨214⟩, ⟨18857⟩⟩
def transferEvent : Nat := 62412
def frameStart : Nat := 62324
def rule : BoundRule := .product (.predecessor 0 62410 .coefficient) (.predecessor 1 62411 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62410 .coefficient)
      LeftAuthority62385.bound (LeftAuthority62385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62385.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62411 .coefficient)
      LeftAuthority62408.bound (LeftAuthority62408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority62385.bound LeftAuthority62408.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62385.bound, LeftAuthority62408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority62385.actual selector witness) * (LeftAuthority62408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62412

namespace LeftBound62420
def owner : Owner := ⟨.program ⟨214⟩, ⟨18862⟩⟩
def transferEvent : Nat := 62420
def frameStart : Nat := 62324
def rule : BoundRule := .sum [.predecessor 0 62418 .coefficient, .predecessor 1 62419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62418 .coefficient)
      LeftAuthority62416.bound (LeftAuthority62416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62416.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62419 .coefficient)
      LeftBound62412.bound (LeftBound62412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62412.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62416.bound, LeftBound62412.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62416.bound, LeftBound62412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62416.actual selector witness, LeftBound62412.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62420

namespace LeftBound62424
def owner : Owner := ⟨.program ⟨214⟩, ⟨28746⟩⟩
def transferEvent : Nat := 62424
def frameStart : Nat := 62324
def rule : BoundRule := .sum [.predecessor 0 62422 .coefficient, .predecessor 1 62423 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62422 .coefficient)
      LeftBound62420.bound (LeftBound62420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62423 .coefficient)
      LeftBound62401.bound (LeftBound62401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62420.bound, LeftBound62401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62420.bound, LeftBound62401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62420.actual selector witness, LeftBound62401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62424

namespace LeftBound62437
def owner : Owner := ⟨.program ⟨214⟩, ⟨28743⟩⟩
def transferEvent : Nat := 62437
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62435 .coefficient, .predecessor 1 62436 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62435 .coefficient)
      LeftBound62266.bound (LeftBound62266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62266.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62436 .coefficient)
      LeftBound62249.bound (LeftBound62249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62266.bound, LeftBound62249.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62266.bound, LeftBound62249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62266.actual selector witness, LeftBound62249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62437

namespace LeftBound62440
def owner : Owner := ⟨.program ⟨214⟩, ⟨28743⟩⟩
def transferEvent : Nat := 62440
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62434 .summary, .result 62256 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62434 .summary)
      LeftBound62268.bound (LeftBound62268.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21911⟩⟩) (rawTerms := some (Proof.Events243.exact62434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62256 .summary)
      LeftBound62251.bound (LeftBound62251.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28742⟩⟩) (rawTerms := some (Proof.Events243.exact62256RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62268.bound, LeftBound62251.bound]
def bound : CoeffClass := .finite ⟨1292270185944771604480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62268.bound, LeftBound62251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62268.actual selector witness, LeftBound62251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62440

namespace LeftBound62444
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def transferEvent : Nat := 62444
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62442 .coefficient) (.predecessor 1 62443 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62442 .coefficient)
      LeftBound62437.bound (LeftBound62437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62443 .coefficient)
      LeftBound5638.bound (LeftBound5638.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5639RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5638.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5638.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62437.bound LeftBound5638.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62437.bound, LeftBound5638.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62437.actual selector witness) * (LeftBound5638.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62444

namespace LeftBound62445
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def transferEvent : Nat := 62445
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6673⟩⟩]⟩ [⟨.result 5635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5635 .coefficient)
      LeftAuthority5634.bound (LeftAuthority5634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6673⟩⟩) (rawTerms := some (Proof.Events022.exact5635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62445

namespace LeftBound62446
def owner : Owner := ⟨.program ⟨214⟩, ⟨28744⟩⟩
def transferEvent : Nat := 62446
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 62441 .summary) (.transfer 62445) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62441 .summary)
      LeftBound62440.bound (LeftBound62440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28743⟩⟩) (rawTerms := some (Proof.Events243.exact62441RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62445)
      LeftBound62445.bound (LeftBound62445.actual selector witness) := by
  exact .transfer (LeftBound62445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62440.bound LeftBound62445.bound
def bound : CoeffClass := .finite ⟨4742652258740286904787271680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62440.bound, LeftBound62445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62440.actual selector witness) * (LeftBound62445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62446

namespace LeftBound62461
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def transferEvent : Nat := 62461
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62459 .coefficient) (.predecessor 1 62460 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62459 .coefficient)
      LeftBound54318.bound (LeftBound54318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62460 .coefficient)
      LeftAuthority62457.bound (LeftAuthority62457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events243.exact62458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62457.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54318.bound LeftAuthority62457.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54318.bound, LeftAuthority62457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54318.actual selector witness) * (LeftAuthority62457.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62461

namespace LeftBound62462
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def transferEvent : Nat := 62462
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28523⟩⟩]⟩ [⟨.result 62458 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62458 .coefficient)
      LeftAuthority62457.bound (LeftAuthority62457.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28523⟩⟩) (rawTerms := some (Proof.Events243.exact62458RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62457.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62457.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62457.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62462

namespace LeftBound62463
def owner : Owner := ⟨.program ⟨214⟩, ⟨28525⟩⟩
def transferEvent : Nat := 62463
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54322 .summary) (.transfer 62462) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54322 .summary)
      LeftBound54321.bound (LeftBound54321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25149⟩⟩) (rawTerms := some (Proof.Events212.exact54322RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62462)
      LeftBound62462.bound (LeftBound62462.actual selector witness) := by
  exact .transfer (LeftBound62462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54321.bound LeftBound62462.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54321.bound, LeftBound62462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54321.actual selector witness) * (LeftBound62462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62463

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
