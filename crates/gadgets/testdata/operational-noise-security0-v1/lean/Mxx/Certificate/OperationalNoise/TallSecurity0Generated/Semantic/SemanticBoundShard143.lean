import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard036
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard142

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22358
def owner : Owner := ⟨.program ⟨214⟩, ⟨29863⟩⟩
def transferEvent : Nat := 22358
def frameStart : Nat := 22258
def rule : BoundRule := .sum [.predecessor 0 22356 .coefficient, .predecessor 1 22357 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22356 .coefficient)
      LeftBound22354.bound (LeftBound22354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22357 .coefficient)
      LeftBound22335.bound (LeftBound22335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22354.bound, LeftBound22335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22354.bound, LeftBound22335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22354.actual selector witness, LeftBound22335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22358

namespace LeftBound22371
def owner : Owner := ⟨.program ⟨214⟩, ⟨29861⟩⟩
def transferEvent : Nat := 22371
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22369 .coefficient, .predecessor 1 22370 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22369 .coefficient)
      LeftBound22200.bound (LeftBound22200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22370 .coefficient)
      LeftBound22183.bound (LeftBound22183.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22183.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22183.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22200.bound, LeftBound22183.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22200.bound, LeftBound22183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22200.actual selector witness, LeftBound22183.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22371

namespace LeftBound22374
def owner : Owner := ⟨.program ⟨214⟩, ⟨29861⟩⟩
def transferEvent : Nat := 22374
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22368 .summary, .result 22190 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22368 .summary)
      LeftBound22202.bound (LeftBound22202.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22711⟩⟩) (rawTerms := some (Proof.Events087.exact22368RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22202.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22190 .summary)
      LeftBound22185.bound (LeftBound22185.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29860⟩⟩) (rawTerms := some (Proof.Events086.exact22190RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22202.bound, LeftBound22185.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22202.bound, LeftBound22185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22202.actual selector witness, LeftBound22185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22374

namespace LeftBound22398
def owner : Owner := ⟨.program ⟨214⟩, ⟨12985⟩⟩
def transferEvent : Nat := 22398
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 22396 .coefficient) (.predecessor 1 22397 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22396 .coefficient)
      LeftAuthority887.bound (LeftAuthority887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22397 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority887.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority887.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority887.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22398

namespace LeftBound22403
def owner : Owner := ⟨.program ⟨214⟩, ⟨7358⟩⟩
def transferEvent : Nat := 22403
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22401 .coefficient) (.predecessor 1 22402 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22401 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22402 .coefficient)
      LeftBound7473.bound (LeftBound7473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound7473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound7473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound7473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22403

namespace LeftBound22408
def owner : Owner := ⟨.program ⟨214⟩, ⟨12986⟩⟩
def transferEvent : Nat := 22408
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22406 .coefficient, .predecessor 1 22407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22406 .coefficient)
      LeftBound22403.bound (LeftBound22403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22407 .coefficient)
      LeftBound22398.bound (LeftBound22398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22403.bound, LeftBound22398.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22403.bound, LeftBound22398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22403.actual selector witness, LeftBound22398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22408

namespace LeftBound22412
def owner : Owner := ⟨.program ⟨214⟩, ⟨12987⟩⟩
def transferEvent : Nat := 22412
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22410 .coefficient, .predecessor 1 22411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22410 .coefficient)
      LeftBound22408.bound (LeftBound22408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22411 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22408.bound, LeftBound7465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22408.bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22408.actual selector witness, LeftBound7465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22412

namespace LeftBound22413
def owner : Owner := ⟨.program ⟨214⟩, ⟨12987⟩⟩
def transferEvent : Nat := 22413
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨102⟩⟩]⟩ [⟨.result 7466 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7466 .coefficient)
      LeftBound7465.bound (LeftBound7465.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events029.exact7466RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7465.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7465.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7465.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22413

namespace LeftBound22418
def owner : Owner := ⟨.program ⟨214⟩, ⟨12988⟩⟩
def transferEvent : Nat := 22418
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22416 .coefficient) (.predecessor 1 22417 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22416 .coefficient)
      LeftBound22412.bound (LeftBound22412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22417 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound22412.bound LeftAuthority890.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22412.bound, LeftAuthority890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound22412.actual selector witness) * (LeftAuthority890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22418

namespace LeftBound22419
def owner : Owner := ⟨.program ⟨214⟩, ⟨12988⟩⟩
def transferEvent : Nat := 22419
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10150⟩⟩], []⟩ [⟨.result 891 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 891 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10150⟩⟩) (rawTerms := some (Proof.Events003.exact891RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority890.bound []
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority890.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22419

namespace LeftBound22420
def owner : Owner := ⟨.program ⟨214⟩, ⟨12988⟩⟩
def transferEvent : Nat := 22420
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22415 .summary) (.transfer 22419) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22415 .summary)
      LeftBound22413.bound (LeftBound22413.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12987⟩⟩) (rawTerms := some (Proof.Events087.exact22415RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22419)
      LeftBound22419.bound (LeftBound22419.actual selector witness) := by
  exact .transfer (LeftBound22419.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound22413.bound LeftBound22419.bound
def bound : CoeffClass := .finite ⟨43264, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22413.bound, LeftBound22419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound22413.actual selector witness) * (LeftBound22419.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22420

namespace LeftBound22426
def owner : Owner := ⟨.program ⟨214⟩, ⟨10151⟩⟩
def transferEvent : Nat := 22426
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 22424 .coefficient) (.predecessor 1 22425 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22424 .coefficient)
      LeftAuthority890.bound (LeftAuthority890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22425 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority890.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority890.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority890.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22426

namespace LeftBound22431
def owner : Owner := ⟨.program ⟨214⟩, ⟨7338⟩⟩
def transferEvent : Nat := 22431
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22429 .coefficient) (.predecessor 1 22430 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22429 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22430 .coefficient)
      LeftBound7514.bound (LeftBound7514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound7514.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound7514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound7514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22431

namespace LeftBound22436
def owner : Owner := ⟨.program ⟨214⟩, ⟨10152⟩⟩
def transferEvent : Nat := 22436
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22434 .coefficient, .predecessor 1 22435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22434 .coefficient)
      LeftBound22431.bound (LeftBound22431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22435 .coefficient)
      LeftBound22426.bound (LeftBound22426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22426.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22431.bound, LeftBound22426.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22431.bound, LeftBound22426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22431.actual selector witness, LeftBound22426.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22436

namespace LeftBound22440
def owner : Owner := ⟨.program ⟨214⟩, ⟨10153⟩⟩
def transferEvent : Nat := 22440
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 22438 .coefficient, .predecessor 1 22439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22438 .coefficient)
      LeftBound22436.bound (LeftBound22436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22439 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22436.bound, LeftBound7506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22436.bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22436.actual selector witness, LeftBound7506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22440

namespace LeftBound22441
def owner : Owner := ⟨.program ⟨214⟩, ⟨10153⟩⟩
def transferEvent : Nat := 22441
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩ [⟨.result 7507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7507 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨82⟩⟩) (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7506.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22441

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
