import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard065
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound40397
def owner : Owner := ⟨.program ⟨214⟩, ⟨11562⟩⟩
def transferEvent : Nat := 40397
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 40395 .coefficient) (.predecessor 1 40396 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40395 .coefficient)
      LeftAuthority1796.bound (LeftAuthority1796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1796.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40396 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1796.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1796.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1796.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40397

namespace LeftBound40402
def owner : Owner := ⟨.program ⟨214⟩, ⟨7312⟩⟩
def transferEvent : Nat := 40402
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40400 .coefficient) (.predecessor 1 40401 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40400 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40401 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40402

namespace LeftBound40407
def owner : Owner := ⟨.program ⟨214⟩, ⟨11563⟩⟩
def transferEvent : Nat := 40407
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40405 .coefficient, .predecessor 1 40406 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40405 .coefficient)
      LeftBound40402.bound (LeftBound40402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40406 .coefficient)
      LeftBound40397.bound (LeftBound40397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40402.bound, LeftBound40397.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40402.bound, LeftBound40397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40402.actual selector witness, LeftBound40397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40407

namespace LeftBound40411
def owner : Owner := ⟨.program ⟨214⟩, ⟨11564⟩⟩
def transferEvent : Nat := 40411
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40409 .coefficient, .predecessor 1 40410 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40409 .coefficient)
      LeftBound40407.bound (LeftBound40407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40410 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40407.bound, LeftBound10972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40407.bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40407.actual selector witness, LeftBound10972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40411

namespace LeftBound40412
def owner : Owner := ⟨.program ⟨214⟩, ⟨11564⟩⟩
def transferEvent : Nat := 40412
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩ [⟨.result 10973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10973 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨94⟩⟩) (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10972.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40412

namespace LeftBound40417
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def transferEvent : Nat := 40417
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40415 .coefficient) (.predecessor 1 40416 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40415 .coefficient)
      LeftBound40411.bound (LeftBound40411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40411.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40416 .coefficient)
      LeftAuthority1799.bound (LeftAuthority1799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1799.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound40411.bound LeftAuthority1799.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40411.bound, LeftAuthority1799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound40411.actual selector witness) * (LeftAuthority1799.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40417

namespace LeftBound40418
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def transferEvent : Nat := 40418
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14442⟩⟩], []⟩ [⟨.result 1800 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1800 .coefficient)
      LeftAuthority1799.bound (LeftAuthority1799.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14442⟩⟩) (rawTerms := some (Proof.Events007.exact1800RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1799.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1799.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1799.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40418

namespace LeftBound40419
def owner : Owner := ⟨.program ⟨214⟩, ⟨14445⟩⟩
def transferEvent : Nat := 40419
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40414 .summary) (.transfer 40418) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40414 .summary)
      LeftBound40412.bound (LeftBound40412.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11564⟩⟩) (rawTerms := some (Proof.Events157.exact40414RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40418)
      LeftBound40418.bound (LeftBound40418.actual selector witness) := by
  exact .transfer (LeftBound40418.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound40412.bound LeftBound40418.bound
def bound : CoeffClass := .finite ⟨18304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40412.bound, LeftBound40418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound40412.actual selector witness) * (LeftBound40418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40419

namespace LeftBound40425
def owner : Owner := ⟨.program ⟨214⟩, ⟨14446⟩⟩
def transferEvent : Nat := 40425
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 40423 .coefficient) (.predecessor 1 40424 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40423 .coefficient)
      LeftAuthority1799.bound (LeftAuthority1799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events007.exact1800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40424 .coefficient)
      LeftBound36043.bound (LeftBound36043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36043.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1799.bound LeftBound36043.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1799.bound, LeftBound36043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1799.actual selector witness) * (LeftBound36043.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound40425

namespace LeftBound40430
def owner : Owner := ⟨.program ⟨214⟩, ⟨7293⟩⟩
def transferEvent : Nat := 40430
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40428 .coefficient) (.predecessor 1 40429 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40428 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40429 .coefficient)
      LeftBound11021.bound (LeftBound11021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftBound11021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftBound11021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftBound11021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40430

namespace LeftBound40435
def owner : Owner := ⟨.program ⟨214⟩, ⟨14447⟩⟩
def transferEvent : Nat := 40435
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40433 .coefficient, .predecessor 1 40434 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40433 .coefficient)
      LeftBound40430.bound (LeftBound40430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40434 .coefficient)
      LeftBound40425.bound (LeftBound40425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40430.bound, LeftBound40425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40430.bound, LeftBound40425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40430.actual selector witness, LeftBound40425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40435

namespace LeftBound40439
def owner : Owner := ⟨.program ⟨214⟩, ⟨14448⟩⟩
def transferEvent : Nat := 40439
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 40437 .coefficient, .predecessor 1 40438 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40437 .coefficient)
      LeftBound40435.bound (LeftBound40435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40435.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40438 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound40435.bound, LeftBound11013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40435.bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound40435.actual selector witness, LeftBound11013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound40439

namespace LeftBound40440
def owner : Owner := ⟨.program ⟨214⟩, ⟨14448⟩⟩
def transferEvent : Nat := 40440
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨75⟩⟩]⟩ [⟨.result 11014 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11014 .coefficient)
      LeftBound11013.bound (LeftBound11013.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨75⟩⟩) (rawTerms := some (Proof.Events043.exact11014RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11013.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound11013.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound11013.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40440

namespace LeftBound40445
def owner : Owner := ⟨.program ⟨214⟩, ⟨14449⟩⟩
def transferEvent : Nat := 40445
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 40443 .coefficient) (.predecessor 1 40444 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 40443 .coefficient)
      LeftBound40439.bound (LeftBound40439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events157.exact40442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 40444 .coefficient)
      LeftBound11010.bound (LeftBound11010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40439.bound LeftBound11010.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40439.bound, LeftBound11010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40439.actual selector witness) * (LeftBound11010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40445

namespace LeftBound40446
def owner : Owner := ⟨.program ⟨214⟩, ⟨14449⟩⟩
def transferEvent : Nat := 40446
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7855⟩⟩]⟩ [⟨.result 11007 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11007 .coefficient)
      LeftAuthority11006.bound (LeftAuthority11006.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7855⟩⟩) (rawTerms := some (Proof.Events042.exact11007RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11006.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11006.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11006.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound40446

namespace LeftBound40447
def owner : Owner := ⟨.program ⟨214⟩, ⟨14449⟩⟩
def transferEvent : Nat := 40447
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40442 .summary) (.transfer 40446) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40442 .summary)
      LeftBound40440.bound (LeftBound40440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14448⟩⟩) (rawTerms := some (Proof.Events157.exact40442RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 40446)
      LeftBound40446.bound (LeftBound40446.actual selector witness) := by
  exact .transfer (LeftBound40446.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40440.bound LeftBound40446.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40440.bound, LeftBound40446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40440.actual selector witness) * (LeftBound40446.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound40447

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
