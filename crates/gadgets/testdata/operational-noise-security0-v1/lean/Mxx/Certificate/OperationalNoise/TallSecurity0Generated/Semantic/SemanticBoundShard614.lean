import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard613

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound90416
def owner : Owner := ⟨.program ⟨214⟩, ⟨22626⟩⟩
def transferEvent : Nat := 90416
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 90414 .coefficient) (.value (.predecessor 1 90415 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90414 .coefficient)
      LeftAuthority90412.bound (LeftAuthority90412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90415 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority90412.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90412.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90412.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound90416

namespace LeftBound90420
def owner : Owner := ⟨.program ⟨214⟩, ⟨22627⟩⟩
def transferEvent : Nat := 90420
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90418 .coefficient) (.predecessor 1 90419 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90418 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90419 .coefficient)
      LeftBound90416.bound (LeftBound90416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90416.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound90416.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound90416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound90416.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90420

namespace LeftBound90421
def owner : Owner := ⟨.program ⟨214⟩, ⟨22627⟩⟩
def transferEvent : Nat := 90421
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22624⟩⟩]⟩ [⟨.result 90413 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90413 .coefficient)
      LeftAuthority90412.bound (LeftAuthority90412.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22624⟩⟩) (rawTerms := some (Proof.Events353.exact90413RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90412.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90412.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority90412.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority90412.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound90421

namespace LeftBound90422
def owner : Owner := ⟨.program ⟨214⟩, ⟨22627⟩⟩
def transferEvent : Nat := 90422
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 90421) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 90421)
      LeftBound90421.bound (LeftBound90421.actual selector witness) := by
  exact .transfer (LeftBound90421.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound90421.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound90421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound90421.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90422

namespace LeftBound90517
def owner : Owner := ⟨.program ⟨214⟩, ⟨16872⟩⟩
def transferEvent : Nat := 90517
def frameStart : Nat := 90478
def rule : BoundRule := .identity (.predecessor 0 90516 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90516 .coefficient)
      LeftAuthority90514.bound (LeftAuthority90514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90514.derived selector witness)

def rawBound : CoeffClass := LeftAuthority90514.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority90514.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90517

namespace LeftBound90534
def owner : Owner := ⟨.program ⟨214⟩, ⟨16967⟩⟩
def transferEvent : Nat := 90534
def frameStart : Nat := 90478
def rule : BoundRule := .sum [.predecessor 0 90532 .coefficient, .predecessor 1 90533 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90532 .coefficient)
      LeftBound90517.bound (LeftBound90517.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90533 .coefficient)
      LeftAuthority90530.bound (LeftAuthority90530.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority90530.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90517.bound, LeftAuthority90530.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90517.bound, LeftAuthority90530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90517.actual selector witness, LeftAuthority90530.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90534

namespace LeftBound90537
def owner : Owner := ⟨.program ⟨214⟩, ⟨16968⟩⟩
def transferEvent : Nat := 90537
def frameStart : Nat := 90478
def rule : BoundRule := .identity (.predecessor 0 90536 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90536 .coefficient)
      LeftBound90534.bound (LeftBound90534.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound90534.derived selector witness)

def rawBound : CoeffClass := LeftBound90534.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound90534.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound90537

namespace LeftBound90543
def owner : Owner := ⟨.program ⟨214⟩, ⟨16969⟩⟩
def transferEvent : Nat := 90543
def frameStart : Nat := 90478
def rule : BoundRule := .product (.predecessor 0 90541 .coefficient) (.predecessor 1 90542 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90541 .coefficient)
      LeftAuthority90539.bound (LeftAuthority90539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90542 .coefficient)
      LeftBound90537.bound (LeftBound90537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority90539.bound LeftBound90537.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90539.bound, LeftBound90537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority90539.actual selector witness) * (LeftBound90537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90543

namespace LeftBound90551
def owner : Owner := ⟨.program ⟨214⟩, ⟨16970⟩⟩
def transferEvent : Nat := 90551
def frameStart : Nat := 90478
def rule : BoundRule := .sum [.predecessor 0 90549 .coefficient, .predecessor 1 90550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90549 .coefficient)
      LeftAuthority90547.bound (LeftAuthority90547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90550 .coefficient)
      LeftBound90543.bound (LeftBound90543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90547.bound, LeftBound90543.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90547.bound, LeftBound90543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90547.actual selector witness, LeftBound90543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90551

namespace LeftBound90555
def owner : Owner := ⟨.program ⟨214⟩, ⟨29813⟩⟩
def transferEvent : Nat := 90555
def frameStart : Nat := 90478
def rule : BoundRule := .product (.predecessor 0 90553 .coefficient) (.predecessor 1 90554 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90553 .coefficient)
      LeftBound90551.bound (LeftBound90551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90551.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90554 .coefficient)
      LeftAuthority90528.bound (LeftAuthority90528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90528.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90551.bound LeftAuthority90528.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90551.bound, LeftAuthority90528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90551.actual selector witness) * (LeftAuthority90528.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90555

namespace LeftBound90566
def owner : Owner := ⟨.program ⟨214⟩, ⟨16929⟩⟩
def transferEvent : Nat := 90566
def frameStart : Nat := 90478
def rule : BoundRule := .product (.predecessor 0 90564 .coefficient) (.predecessor 1 90565 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90564 .coefficient)
      LeftAuthority90539.bound (LeftAuthority90539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90565 .coefficient)
      LeftAuthority90562.bound (LeftAuthority90562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority90539.bound LeftAuthority90562.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90539.bound, LeftAuthority90562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority90539.actual selector witness) * (LeftAuthority90562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90566

namespace LeftBound90574
def owner : Owner := ⟨.program ⟨214⟩, ⟨16930⟩⟩
def transferEvent : Nat := 90574
def frameStart : Nat := 90478
def rule : BoundRule := .sum [.predecessor 0 90572 .coefficient, .predecessor 1 90573 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90572 .coefficient)
      LeftAuthority90570.bound (LeftAuthority90570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority90570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority90570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90573 .coefficient)
      LeftBound90566.bound (LeftBound90566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority90570.bound, LeftBound90566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority90570.bound, LeftBound90566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority90570.actual selector witness, LeftBound90566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90574

namespace LeftBound90578
def owner : Owner := ⟨.program ⟨214⟩, ⟨29818⟩⟩
def transferEvent : Nat := 90578
def frameStart : Nat := 90478
def rule : BoundRule := .sum [.predecessor 0 90576 .coefficient, .predecessor 1 90577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90576 .coefficient)
      LeftBound90574.bound (LeftBound90574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90577 .coefficient)
      LeftBound90555.bound (LeftBound90555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90555.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90574.bound, LeftBound90555.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90574.bound, LeftBound90555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90574.actual selector witness, LeftBound90555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90578

namespace LeftBound90591
def owner : Owner := ⟨.program ⟨214⟩, ⟨29815⟩⟩
def transferEvent : Nat := 90591
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 90589 .coefficient, .predecessor 1 90590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90589 .coefficient)
      LeftBound90420.bound (LeftBound90420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90590 .coefficient)
      LeftBound90403.bound (LeftBound90403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90403.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90420.bound, LeftBound90403.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90420.bound, LeftBound90403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90420.actual selector witness, LeftBound90403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90591

namespace LeftBound90594
def owner : Owner := ⟨.program ⟨214⟩, ⟨29815⟩⟩
def transferEvent : Nat := 90594
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 90588 .summary, .result 90410 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90588 .summary)
      LeftBound90422.bound (LeftBound90422.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22627⟩⟩) (rawTerms := some (Proof.Events353.exact90588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 90410 .summary)
      LeftBound90405.bound (LeftBound90405.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29814⟩⟩) (rawTerms := some (Proof.Events353.exact90410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound90405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound90422.bound, LeftBound90405.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90422.bound, LeftBound90405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound90422.actual selector witness, LeftBound90405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound90594

namespace LeftBound90598
def owner : Owner := ⟨.program ⟨214⟩, ⟨29816⟩⟩
def transferEvent : Nat := 90598
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 90596 .coefficient) (.predecessor 1 90597 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 90596 .coefficient)
      LeftBound90591.bound (LeftBound90591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 90597 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound90591.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90591.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound90591.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound90598

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
