import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard283

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42250
def owner : Owner := ⟨.program ⟨214⟩, ⟨15787⟩⟩
def transferEvent : Nat := 42250
def frameStart : Nat := 42185
def rule : BoundRule := .product (.predecessor 0 42248 .coefficient) (.predecessor 1 42249 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42248 .coefficient)
      LeftAuthority42246.bound (LeftAuthority42246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42249 .coefficient)
      LeftBound42244.bound (LeftBound42244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority42246.bound LeftBound42244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42246.bound, LeftBound42244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority42246.actual selector witness) * (LeftBound42244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42250

namespace LeftBound42258
def owner : Owner := ⟨.program ⟨214⟩, ⟨15788⟩⟩
def transferEvent : Nat := 42258
def frameStart : Nat := 42185
def rule : BoundRule := .sum [.predecessor 0 42256 .coefficient, .predecessor 1 42257 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42256 .coefficient)
      LeftAuthority42254.bound (LeftAuthority42254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42254.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42254.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42257 .coefficient)
      LeftBound42250.bound (LeftBound42250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42250.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42254.bound, LeftBound42250.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42254.bound, LeftBound42250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42254.actual selector witness, LeftBound42250.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42258

namespace LeftBound42262
def owner : Owner := ⟨.program ⟨214⟩, ⟨27459⟩⟩
def transferEvent : Nat := 42262
def frameStart : Nat := 42185
def rule : BoundRule := .product (.predecessor 0 42260 .coefficient) (.predecessor 1 42261 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42260 .coefficient)
      LeftBound42258.bound (LeftBound42258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42261 .coefficient)
      LeftAuthority42235.bound (LeftAuthority42235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42258.bound LeftAuthority42235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42258.bound, LeftAuthority42235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42258.actual selector witness) * (LeftAuthority42235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42262

namespace LeftBound42273
def owner : Owner := ⟨.program ⟨214⟩, ⟨15755⟩⟩
def transferEvent : Nat := 42273
def frameStart : Nat := 42185
def rule : BoundRule := .product (.predecessor 0 42271 .coefficient) (.predecessor 1 42272 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42271 .coefficient)
      LeftAuthority42246.bound (LeftAuthority42246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42272 .coefficient)
      LeftAuthority42269.bound (LeftAuthority42269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42246.bound LeftAuthority42269.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42246.bound, LeftAuthority42269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42246.actual selector witness) * (LeftAuthority42269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42273

namespace LeftBound42281
def owner : Owner := ⟨.program ⟨214⟩, ⟨15756⟩⟩
def transferEvent : Nat := 42281
def frameStart : Nat := 42185
def rule : BoundRule := .sum [.predecessor 0 42279 .coefficient, .predecessor 1 42280 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42279 .coefficient)
      LeftAuthority42277.bound (LeftAuthority42277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42280 .coefficient)
      LeftBound42273.bound (LeftBound42273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority42277.bound, LeftBound42273.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42277.bound, LeftBound42273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority42277.actual selector witness, LeftBound42273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42281

namespace LeftBound42285
def owner : Owner := ⟨.program ⟨214⟩, ⟨27463⟩⟩
def transferEvent : Nat := 42285
def frameStart : Nat := 42185
def rule : BoundRule := .sum [.predecessor 0 42283 .coefficient, .predecessor 1 42284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42283 .coefficient)
      LeftBound42281.bound (LeftBound42281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42284 .coefficient)
      LeftBound42262.bound (LeftBound42262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42262.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42281.bound, LeftBound42262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42281.bound, LeftBound42262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42281.actual selector witness, LeftBound42262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42285

namespace LeftBound42298
def owner : Owner := ⟨.program ⟨214⟩, ⟨27461⟩⟩
def transferEvent : Nat := 42298
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42296 .coefficient, .predecessor 1 42297 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42296 .coefficient)
      LeftBound42127.bound (LeftBound42127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42297 .coefficient)
      LeftBound42110.bound (LeftBound42110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42127.bound, LeftBound42110.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42127.bound, LeftBound42110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42127.actual selector witness, LeftBound42110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42298

namespace LeftBound42301
def owner : Owner := ⟨.program ⟨214⟩, ⟨27461⟩⟩
def transferEvent : Nat := 42301
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 42295 .summary, .result 42117 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42295 .summary)
      LeftBound42129.bound (LeftBound42129.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21123⟩⟩) (rawTerms := some (Proof.Events165.exact42295RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42117 .summary)
      LeftBound42112.bound (LeftBound42112.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27460⟩⟩) (rawTerms := some (Proof.Events164.exact42117RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42129.bound, LeftBound42112.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42129.bound, LeftBound42112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42129.actual selector witness, LeftBound42112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42301

namespace LeftBound42325
def owner : Owner := ⟨.program ⟨214⟩, ⟨11226⟩⟩
def transferEvent : Nat := 42325
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 42323 .coefficient) (.predecessor 1 42324 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42323 .coefficient)
      LeftAuthority1888.bound (LeftAuthority1888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42324 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1888.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1888.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1888.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42325

namespace LeftBound42330
def owner : Owner := ⟨.program ⟨214⟩, ⟨7308⟩⟩
def transferEvent : Nat := 42330
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42328 .coefficient) (.predecessor 1 42329 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42328 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42329 .coefficient)
      LeftBound12984.bound (LeftBound12984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound12984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound12984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound12984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42330

namespace LeftBound42335
def owner : Owner := ⟨.program ⟨214⟩, ⟨11227⟩⟩
def transferEvent : Nat := 42335
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42333 .coefficient, .predecessor 1 42334 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42333 .coefficient)
      LeftBound42330.bound (LeftBound42330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42334 .coefficient)
      LeftBound42325.bound (LeftBound42325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42330.bound, LeftBound42325.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42330.bound, LeftBound42325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42330.actual selector witness, LeftBound42325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42335

namespace LeftBound42339
def owner : Owner := ⟨.program ⟨214⟩, ⟨11228⟩⟩
def transferEvent : Nat := 42339
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42337 .coefficient, .predecessor 1 42338 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42337 .coefficient)
      LeftBound42335.bound (LeftBound42335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42338 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42335.bound, LeftBound12976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42335.bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42335.actual selector witness, LeftBound12976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42339

namespace LeftBound42340
def owner : Owner := ⟨.program ⟨214⟩, ⟨11228⟩⟩
def transferEvent : Nat := 42340
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩ [⟨.result 12977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12977 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨90⟩⟩) (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12976.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42340

namespace LeftBound42345
def owner : Owner := ⟨.program ⟨214⟩, ⟨13577⟩⟩
def transferEvent : Nat := 42345
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42343 .coefficient) (.predecessor 1 42344 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42343 .coefficient)
      LeftBound42339.bound (LeftBound42339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42344 .coefficient)
      LeftAuthority1891.bound (LeftAuthority1891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound42339.bound LeftAuthority1891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42339.bound, LeftAuthority1891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound42339.actual selector witness) * (LeftAuthority1891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42345

namespace LeftBound42346
def owner : Owner := ⟨.program ⟨214⟩, ⟨13577⟩⟩
def transferEvent : Nat := 42346
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13574⟩⟩], []⟩ [⟨.result 1892 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1892 .coefficient)
      LeftAuthority1891.bound (LeftAuthority1891.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13574⟩⟩) (rawTerms := some (Proof.Events007.exact1892RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1891.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1891.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1891.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42346

namespace LeftBound42347
def owner : Owner := ⟨.program ⟨214⟩, ⟨13577⟩⟩
def transferEvent : Nat := 42347
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42342 .summary) (.transfer 42346) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42342 .summary)
      LeftBound42340.bound (LeftBound42340.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11228⟩⟩) (rawTerms := some (Proof.Events165.exact42342RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42346)
      LeftBound42346.bound (LeftBound42346.actual selector witness) := by
  exact .transfer (LeftBound42346.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound42340.bound LeftBound42346.bound
def bound : CoeffClass := .finite ⟨8320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42340.bound, LeftBound42346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound42340.actual selector witness) * (LeftBound42346.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42347

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
