import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard284

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound42353
def owner : Owner := ⟨.program ⟨214⟩, ⟨13578⟩⟩
def transferEvent : Nat := 42353
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 42351 .coefficient) (.predecessor 1 42352 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42351 .coefficient)
      LeftAuthority1891.bound (LeftAuthority1891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42352 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1891.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1891.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1891.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42353

namespace LeftBound42358
def owner : Owner := ⟨.program ⟨214⟩, ⟨7325⟩⟩
def transferEvent : Nat := 42358
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42356 .coefficient) (.predecessor 1 42357 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42356 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42357 .coefficient)
      LeftBound13025.bound (LeftBound13025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound13025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound13025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound13025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42358

namespace LeftBound42363
def owner : Owner := ⟨.program ⟨214⟩, ⟨13579⟩⟩
def transferEvent : Nat := 42363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42361 .coefficient, .predecessor 1 42362 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42361 .coefficient)
      LeftBound42358.bound (LeftBound42358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42362 .coefficient)
      LeftBound42353.bound (LeftBound42353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42358.bound, LeftBound42353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42358.bound, LeftBound42353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42358.actual selector witness, LeftBound42353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42363

namespace LeftBound42367
def owner : Owner := ⟨.program ⟨214⟩, ⟨13580⟩⟩
def transferEvent : Nat := 42367
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42365 .coefficient, .predecessor 1 42366 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42365 .coefficient)
      LeftBound42363.bound (LeftBound42363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42366 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42363.bound, LeftBound13017.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42363.bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42363.actual selector witness, LeftBound13017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42367

namespace LeftBound42368
def owner : Owner := ⟨.program ⟨214⟩, ⟨13580⟩⟩
def transferEvent : Nat := 42368
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨107⟩⟩]⟩ [⟨.result 13018 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13018 .coefficient)
      LeftBound13017.bound (LeftBound13017.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨107⟩⟩) (rawTerms := some (Proof.Events050.exact13018RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13017.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13017.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13017.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42368

namespace LeftBound42373
def owner : Owner := ⟨.program ⟨214⟩, ⟨13581⟩⟩
def transferEvent : Nat := 42373
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42371 .coefficient) (.predecessor 1 42372 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42371 .coefficient)
      LeftBound42367.bound (LeftBound42367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42372 .coefficient)
      LeftBound13014.bound (LeftBound13014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13014.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42367.bound LeftBound13014.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42367.bound, LeftBound13014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42367.actual selector witness) * (LeftBound13014.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42373

namespace LeftBound42374
def owner : Owner := ⟨.program ⟨214⟩, ⟨13581⟩⟩
def transferEvent : Nat := 42374
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7843⟩⟩]⟩ [⟨.result 13011 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13011 .coefficient)
      LeftAuthority13010.bound (LeftAuthority13010.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7843⟩⟩) (rawTerms := some (Proof.Events050.exact13011RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13010.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13010.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13010.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42374

namespace LeftBound42375
def owner : Owner := ⟨.program ⟨214⟩, ⟨13581⟩⟩
def transferEvent : Nat := 42375
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42370 .summary) (.transfer 42374) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42370 .summary)
      LeftBound42368.bound (LeftBound42368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13580⟩⟩) (rawTerms := some (Proof.Events165.exact42370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42374)
      LeftBound42374.bound (LeftBound42374.actual selector witness) := by
  exact .transfer (LeftBound42374.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42368.bound LeftBound42374.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42368.bound, LeftBound42374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42368.actual selector witness) * (LeftBound42374.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42375

namespace LeftBound42383
def owner : Owner := ⟨.program ⟨214⟩, ⟨13582⟩⟩
def transferEvent : Nat := 42383
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 42381 .coefficient, .predecessor 1 42382 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42381 .coefficient)
      LeftBound42373.bound (LeftBound42373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42382 .coefficient)
      LeftBound42345.bound (LeftBound42345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42345.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42373.bound, LeftBound42345.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42373.bound, LeftBound42345.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42373.actual selector witness, LeftBound42345.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42383

namespace LeftBound42385
def owner : Owner := ⟨.program ⟨214⟩, ⟨13582⟩⟩
def transferEvent : Nat := 42385
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 42380 .summary, .result 42350 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42380 .summary)
      LeftBound42375.bound (LeftBound42375.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13581⟩⟩) (rawTerms := some (Proof.Events165.exact42380RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42350 .summary)
      LeftBound42347.bound (LeftBound42347.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13577⟩⟩) (rawTerms := some (Proof.Events165.exact42350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42347.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42375.bound, LeftBound42347.bound]
def bound : CoeffClass := .finite ⟨95428736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42375.bound, LeftBound42347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42375.actual selector witness, LeftBound42347.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42385

namespace LeftBound42389
def owner : Owner := ⟨.program ⟨214⟩, ⟨25846⟩⟩
def transferEvent : Nat := 42389
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42387 .coefficient) (.predecessor 1 42388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42387 .coefficient)
      LeftBound42383.bound (LeftBound42383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42388 .coefficient)
      LeftAuthority42321.bound (LeftAuthority42321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42321.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42321.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42383.bound LeftAuthority42321.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42383.bound, LeftAuthority42321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42383.actual selector witness) * (LeftAuthority42321.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42389

namespace LeftBound42390
def owner : Owner := ⟨.program ⟨214⟩, ⟨25846⟩⟩
def transferEvent : Nat := 42390
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25845⟩⟩]⟩ [⟨.result 42322 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42322 .coefficient)
      LeftAuthority42321.bound (LeftAuthority42321.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25845⟩⟩) (rawTerms := some (Proof.Events165.exact42322RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42321.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42321.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42321.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42321.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42390

namespace LeftBound42391
def owner : Owner := ⟨.program ⟨214⟩, ⟨25846⟩⟩
def transferEvent : Nat := 42391
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42386 .summary) (.transfer 42390) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42386 .summary)
      LeftBound42385.bound (LeftBound42385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13582⟩⟩) (rawTerms := some (Proof.Events165.exact42386RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 42390)
      LeftBound42390.bound (LeftBound42390.actual selector witness) := by
  exact .transfer (LeftBound42390.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42385.bound LeftBound42390.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42385.bound, LeftBound42390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42385.actual selector witness) * (LeftBound42390.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42391

namespace LeftBound42402
def owner : Owner := ⟨.program ⟨214⟩, ⟨19322⟩⟩
def transferEvent : Nat := 42402
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 42400 .coefficient) (.value (.predecessor 1 42401 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42400 .coefficient)
      LeftAuthority42398.bound (LeftAuthority42398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42401 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority42398.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42398.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42398.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42402

namespace LeftBound42406
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def transferEvent : Nat := 42406
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 42404 .coefficient) (.predecessor 1 42405 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42404 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42405 .coefficient)
      LeftBound42402.bound (LeftBound42402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events165.exact42403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42402.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound42402.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound42402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound42402.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42406

namespace LeftBound42407
def owner : Owner := ⟨.program ⟨214⟩, ⟨19323⟩⟩
def transferEvent : Nat := 42407
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19320⟩⟩]⟩ [⟨.result 42399 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42399 .coefficient)
      LeftAuthority42398.bound (LeftAuthority42398.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19320⟩⟩) (rawTerms := some (Proof.Events165.exact42399RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42398.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority42398.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42398.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound42407

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
