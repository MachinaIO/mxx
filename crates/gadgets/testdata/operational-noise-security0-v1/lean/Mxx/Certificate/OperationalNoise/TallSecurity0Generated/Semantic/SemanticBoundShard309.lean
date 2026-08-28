import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard243
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard308

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46515
def owner : Owner := ⟨.program ⟨214⟩, ⟨18134⟩⟩
def transferEvent : Nat := 46515
def frameStart : Nat := 46427
def rule : BoundRule := .product (.predecessor 0 46513 .coefficient) (.predecessor 1 46514 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46513 .coefficient)
      LeftAuthority46488.bound (LeftAuthority46488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46514 .coefficient)
      LeftAuthority46511.bound (LeftAuthority46511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46511.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46511.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority46488.bound LeftAuthority46511.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46488.bound, LeftAuthority46511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority46488.actual selector witness) * (LeftAuthority46511.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46515

namespace LeftBound46523
def owner : Owner := ⟨.program ⟨214⟩, ⟨18135⟩⟩
def transferEvent : Nat := 46523
def frameStart : Nat := 46427
def rule : BoundRule := .sum [.predecessor 0 46521 .coefficient, .predecessor 1 46522 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46521 .coefficient)
      LeftAuthority46519.bound (LeftAuthority46519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46522 .coefficient)
      LeftBound46515.bound (LeftBound46515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46515.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46519.bound, LeftBound46515.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46519.bound, LeftBound46515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46519.actual selector witness, LeftBound46515.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46523

namespace LeftBound46527
def owner : Owner := ⟨.program ⟨214⟩, ⟨30160⟩⟩
def transferEvent : Nat := 46527
def frameStart : Nat := 46427
def rule : BoundRule := .sum [.predecessor 0 46525 .coefficient, .predecessor 1 46526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46525 .coefficient)
      LeftBound46523.bound (LeftBound46523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46526 .coefficient)
      LeftBound46504.bound (LeftBound46504.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46504.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46504.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46523.bound, LeftBound46504.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46523.bound, LeftBound46504.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46523.actual selector witness, LeftBound46504.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46527

namespace LeftBound46540
def owner : Owner := ⟨.program ⟨214⟩, ⟨30157⟩⟩
def transferEvent : Nat := 46540
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46538 .coefficient, .predecessor 1 46539 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46538 .coefficient)
      LeftBound46369.bound (LeftBound46369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46539 .coefficient)
      LeftBound46352.bound (LeftBound46352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46352.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46369.bound, LeftBound46352.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46369.bound, LeftBound46352.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46369.actual selector witness, LeftBound46352.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46540

namespace LeftBound46543
def owner : Owner := ⟨.program ⟨214⟩, ⟨30157⟩⟩
def transferEvent : Nat := 46543
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46537 .summary, .result 46359 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46537 .summary)
      LeftBound46371.bound (LeftBound46371.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22779⟩⟩) (rawTerms := some (Proof.Events181.exact46537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46359 .summary)
      LeftBound46354.bound (LeftBound46354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30156⟩⟩) (rawTerms := some (Proof.Events181.exact46359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46371.bound, LeftBound46354.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46371.bound, LeftBound46354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46371.actual selector witness, LeftBound46354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46543

namespace LeftBound46547
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def transferEvent : Nat := 46547
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46545 .coefficient) (.predecessor 1 46546 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46545 .coefficient)
      LeftBound46540.bound (LeftBound46540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46546 .coefficient)
      LeftBound5518.bound (LeftBound5518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5518.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46540.bound LeftBound5518.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46540.bound, LeftBound5518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46540.actual selector witness) * (LeftBound5518.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46547

namespace LeftBound46548
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def transferEvent : Nat := 46548
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6657⟩⟩]⟩ [⟨.result 5515 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5515 .coefficient)
      LeftAuthority5514.bound (LeftAuthority5514.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6657⟩⟩) (rawTerms := some (Proof.Events021.exact5515RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5514.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5514.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5514.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46548

namespace LeftBound46549
def owner : Owner := ⟨.program ⟨214⟩, ⟨30158⟩⟩
def transferEvent : Nat := 46549
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46544 .summary) (.transfer 46548) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46544 .summary)
      LeftBound46543.bound (LeftBound46543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30157⟩⟩) (rawTerms := some (Proof.Events181.exact46544RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46548)
      LeftBound46548.bound (LeftBound46548.actual selector witness) := by
  exact .transfer (LeftBound46548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46543.bound LeftBound46548.bound
def bound : CoeffClass := .finite ⟨4743639307122182955475140608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46543.bound, LeftBound46548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46543.actual selector witness) * (LeftBound46548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46549

namespace LeftBound46564
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def transferEvent : Nat := 46564
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46562 .coefficient) (.predecessor 1 46563 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46562 .coefficient)
      LeftBound36801.bound (LeftBound36801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46563 .coefficient)
      LeftAuthority46560.bound (LeftAuthority46560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36801.bound LeftAuthority46560.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36801.bound, LeftAuthority46560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36801.actual selector witness) * (LeftAuthority46560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46564

namespace LeftBound46565
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def transferEvent : Nat := 46565
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29838⟩⟩]⟩ [⟨.result 46561 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46561 .coefficient)
      LeftAuthority46560.bound (LeftAuthority46560.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29838⟩⟩) (rawTerms := some (Proof.Events181.exact46561RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46560.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46560.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46560.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46565

namespace LeftBound46566
def owner : Owner := ⟨.program ⟨214⟩, ⟨29840⟩⟩
def transferEvent : Nat := 46566
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36805 .summary) (.transfer 46565) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36805 .summary)
      LeftBound36804.bound (LeftBound36804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25693⟩⟩) (rawTerms := some (Proof.Events143.exact36805RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46565)
      LeftBound46565.bound (LeftBound46565.actual selector witness) := by
  exact .transfer (LeftBound46565.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36804.bound LeftBound46565.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36804.bound, LeftBound46565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36804.actual selector witness) * (LeftBound46565.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46566

namespace LeftBound46577
def owner : Owner := ⟨.program ⟨214⟩, ⟨22634⟩⟩
def transferEvent : Nat := 46577
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 46575 .coefficient) (.value (.predecessor 1 46576 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46575 .coefficient)
      LeftAuthority46573.bound (LeftAuthority46573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46576 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority46573.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46573.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46573.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound46577

namespace LeftBound46581
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def transferEvent : Nat := 46581
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46579 .coefficient) (.predecessor 1 46580 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46579 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46580 .coefficient)
      LeftBound46577.bound (LeftBound46577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46577.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound46577.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound46577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound46577.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46581

namespace LeftBound46582
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def transferEvent : Nat := 46582
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22632⟩⟩]⟩ [⟨.result 46574 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46574 .coefficient)
      LeftAuthority46573.bound (LeftAuthority46573.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22632⟩⟩) (rawTerms := some (Proof.Events181.exact46574RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46573.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46573.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46573.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46582

namespace LeftBound46583
def owner : Owner := ⟨.program ⟨214⟩, ⟨22635⟩⟩
def transferEvent : Nat := 46583
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 46582) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46582)
      LeftBound46582.bound (LeftBound46582.actual selector witness) := by
  exact .transfer (LeftBound46582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound46582.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound46582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound46582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46583

namespace LeftBound46678
def owner : Owner := ⟨.program ⟨214⟩, ⟨16880⟩⟩
def transferEvent : Nat := 46678
def frameStart : Nat := 46639
def rule : BoundRule := .identity (.predecessor 0 46677 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46677 .coefficient)
      LeftAuthority46675.bound (LeftAuthority46675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46675.derived selector witness)

def rawBound : CoeffClass := LeftAuthority46675.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority46675.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46678

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
