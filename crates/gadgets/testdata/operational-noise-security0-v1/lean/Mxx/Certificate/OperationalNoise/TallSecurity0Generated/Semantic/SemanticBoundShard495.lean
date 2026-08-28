import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard494

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72560
def owner : Owner := ⟨.program ⟨214⟩, ⟨10975⟩⟩
def transferEvent : Nat := 72560
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10837⟩⟩], []⟩ [⟨.result 3434 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3434 .coefficient)
      LeftAuthority3433.bound (LeftAuthority3433.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10837⟩⟩) (rawTerms := some (Proof.Events013.exact3434RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3433.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3433.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3433.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72560

namespace LeftBound72561
def owner : Owner := ⟨.program ⟨214⟩, ⟨10975⟩⟩
def transferEvent : Nat := 72561
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72556 .summary) (.transfer 72560) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72556 .summary)
      LeftBound72554.bound (LeftBound72554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10974⟩⟩) (rawTerms := some (Proof.Events283.exact72556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72560)
      LeftBound72560.bound (LeftBound72560.actual selector witness) := by
  exact .transfer (LeftBound72560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound72554.bound LeftBound72560.bound
def bound : CoeffClass := .finite ⟨3328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72554.bound, LeftBound72560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound72554.actual selector witness) * (LeftBound72560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72561

namespace LeftBound72567
def owner : Owner := ⟨.program ⟨214⟩, ⟨10838⟩⟩
def transferEvent : Nat := 72567
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 72565 .coefficient) (.predecessor 1 72566 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72565 .coefficient)
      LeftAuthority3433.bound (LeftAuthority3433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72566 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3433.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3433.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3433.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72567

namespace LeftBound72572
def owner : Owner := ⟨.program ⟨214⟩, ⟨7209⟩⟩
def transferEvent : Nat := 72572
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72570 .coefficient) (.predecessor 1 72571 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72570 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72571 .coefficient)
      LeftBound14027.bound (LeftBound14027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound14027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound14027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound14027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72572

namespace LeftBound72577
def owner : Owner := ⟨.program ⟨214⟩, ⟨10839⟩⟩
def transferEvent : Nat := 72577
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72575 .coefficient, .predecessor 1 72576 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72575 .coefficient)
      LeftBound72572.bound (LeftBound72572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72576 .coefficient)
      LeftBound72567.bound (LeftBound72567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72572.bound, LeftBound72567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72572.bound, LeftBound72567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72572.actual selector witness, LeftBound72567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72577

namespace LeftBound72581
def owner : Owner := ⟨.program ⟨214⟩, ⟨10840⟩⟩
def transferEvent : Nat := 72581
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72579 .coefficient, .predecessor 1 72580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72579 .coefficient)
      LeftBound72577.bound (LeftBound72577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72580 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72577.bound, LeftBound14019.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72577.bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72577.actual selector witness, LeftBound14019.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72581

namespace LeftBound72582
def owner : Owner := ⟨.program ⟨214⟩, ⟨10840⟩⟩
def transferEvent : Nat := 72582
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨105⟩⟩]⟩ [⟨.result 14020 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14020 .coefficient)
      LeftBound14019.bound (LeftBound14019.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events054.exact14020RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14019.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound14019.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound14019.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72582

namespace LeftBound72587
def owner : Owner := ⟨.program ⟨214⟩, ⟨10841⟩⟩
def transferEvent : Nat := 72587
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72585 .coefficient) (.predecessor 1 72586 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72585 .coefficient)
      LeftBound72581.bound (LeftBound72581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72586 .coefficient)
      LeftBound14016.bound (LeftBound14016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72581.bound LeftBound14016.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72581.bound, LeftBound14016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72581.actual selector witness) * (LeftBound14016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72587

namespace LeftBound72588
def owner : Owner := ⟨.program ⟨214⟩, ⟨10841⟩⟩
def transferEvent : Nat := 72588
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7837⟩⟩]⟩ [⟨.result 14013 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14013 .coefficient)
      LeftAuthority14012.bound (LeftAuthority14012.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7837⟩⟩) (rawTerms := some (Proof.Events054.exact14013RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14012.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14012.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14012.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72588

namespace LeftBound72589
def owner : Owner := ⟨.program ⟨214⟩, ⟨10841⟩⟩
def transferEvent : Nat := 72589
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72584 .summary) (.transfer 72588) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72584 .summary)
      LeftBound72582.bound (LeftBound72582.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10840⟩⟩) (rawTerms := some (Proof.Events283.exact72584RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72588)
      LeftBound72588.bound (LeftBound72588.actual selector witness) := by
  exact .transfer (LeftBound72588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72582.bound LeftBound72588.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72582.bound, LeftBound72588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72582.actual selector witness) * (LeftBound72588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72589

namespace LeftBound72597
def owner : Owner := ⟨.program ⟨214⟩, ⟨10976⟩⟩
def transferEvent : Nat := 72597
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72595 .coefficient, .predecessor 1 72596 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72595 .coefficient)
      LeftBound72587.bound (LeftBound72587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72596 .coefficient)
      LeftBound72559.bound (LeftBound72559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72587.bound, LeftBound72559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72587.bound, LeftBound72559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72587.actual selector witness, LeftBound72559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72597

namespace LeftBound72599
def owner : Owner := ⟨.program ⟨214⟩, ⟨10976⟩⟩
def transferEvent : Nat := 72599
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72594 .summary, .result 72564 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72594 .summary)
      LeftBound72589.bound (LeftBound72589.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10841⟩⟩) (rawTerms := some (Proof.Events283.exact72594RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72564 .summary)
      LeftBound72561.bound (LeftBound72561.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10975⟩⟩) (rawTerms := some (Proof.Events283.exact72564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72589.bound, LeftBound72561.bound]
def bound : CoeffClass := .finite ⟨95423744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72589.bound, LeftBound72561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72589.actual selector witness, LeftBound72561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72599

namespace LeftBound72603
def owner : Owner := ⟨.program ⟨214⟩, ⟨25061⟩⟩
def transferEvent : Nat := 72603
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72601 .coefficient) (.predecessor 1 72602 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72601 .coefficient)
      LeftBound72597.bound (LeftBound72597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72602 .coefficient)
      LeftAuthority72535.bound (LeftAuthority72535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72535.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72597.bound LeftAuthority72535.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72597.bound, LeftAuthority72535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72597.actual selector witness) * (LeftAuthority72535.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72603

namespace LeftBound72604
def owner : Owner := ⟨.program ⟨214⟩, ⟨25061⟩⟩
def transferEvent : Nat := 72604
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25060⟩⟩]⟩ [⟨.result 72536 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72536 .coefficient)
      LeftAuthority72535.bound (LeftAuthority72535.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25060⟩⟩) (rawTerms := some (Proof.Events283.exact72536RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72535.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72535.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72535.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72604

namespace LeftBound72605
def owner : Owner := ⟨.program ⟨214⟩, ⟨25061⟩⟩
def transferEvent : Nat := 72605
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72600 .summary) (.transfer 72604) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72600 .summary)
      LeftBound72599.bound (LeftBound72599.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10976⟩⟩) (rawTerms := some (Proof.Events283.exact72600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72604)
      LeftBound72604.bound (LeftBound72604.actual selector witness) := by
  exact .transfer (LeftBound72604.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72599.bound LeftBound72604.bound
def bound : CoeffClass := .finite ⟨350206667259904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72599.bound, LeftBound72604.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72599.actual selector witness) * (LeftBound72604.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72605

namespace LeftBound72616
def owner : Owner := ⟨.program ⟨214⟩, ⟨19166⟩⟩
def transferEvent : Nat := 72616
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 72614 .coefficient) (.value (.predecessor 1 72615 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72614 .coefficient)
      LeftAuthority72612.bound (LeftAuthority72612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72615 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72612.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72612.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72612.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72616

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
