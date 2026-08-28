import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard085

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13683
def owner : Owner := ⟨.program ⟨214⟩, ⟨12287⟩⟩
def transferEvent : Nat := 13683
def frameStart : Nat := 13630
def rule : BoundRule := .identity (.predecessor 0 13682 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13682 .coefficient)
      LeftBound13680.bound (LeftBound13680.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound13680.derived selector witness)

def rawBound : CoeffClass := LeftBound13680.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound13680.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13683

namespace LeftBound13689
def owner : Owner := ⟨.program ⟨214⟩, ⟨12288⟩⟩
def transferEvent : Nat := 13689
def frameStart : Nat := 13630
def rule : BoundRule := .product (.predecessor 0 13687 .coefficient) (.predecessor 1 13688 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13687 .coefficient)
      LeftAuthority13685.bound (LeftAuthority13685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13688 .coefficient)
      LeftBound13683.bound (LeftBound13683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13683.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority13685.bound LeftBound13683.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13685.bound, LeftBound13683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority13685.actual selector witness) * (LeftBound13683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13689

namespace LeftBound13705
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 13705
def frameStart : Nat := 13630
def rule : BoundRule := .scale (.predecessor 0 13703 .coefficient) (.value (.predecessor 1 13704 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13703 .coefficient)
      LeftAuthority13701.bound (LeftAuthority13701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13704 .coefficient)
      LeftAuthority13692.bound (LeftAuthority13692.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority13692.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13701.bound LeftAuthority13692.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13701.bound, LeftAuthority13692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13701.actual selector witness) * (LeftAuthority13692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13705

namespace LeftBound13708
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 13708
def frameStart : Nat := 13630
def rule : BoundRule := .identity (.predecessor 0 13707 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13707 .coefficient)
      LeftAuthority13695.bound (LeftAuthority13695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13695.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13695.derived selector witness)

def rawBound : CoeffClass := LeftAuthority13695.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority13695.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound13708

namespace LeftBound13712
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 13712
def frameStart : Nat := 13630
def rule : BoundRule := .product (.predecessor 0 13710 .coefficient) (.predecessor 1 13711 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13710 .coefficient)
      LeftBound13708.bound (LeftBound13708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13711 .coefficient)
      LeftBound13705.bound (LeftBound13705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13705.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13708.bound LeftBound13705.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13708.bound, LeftBound13705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13708.actual selector witness) * (LeftBound13705.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13712

namespace LeftBound13717
def owner : Owner := ⟨.program ⟨214⟩, ⟨12289⟩⟩
def transferEvent : Nat := 13717
def frameStart : Nat := 13630
def rule : BoundRule := .sum [.predecessor 0 13715 .coefficient, .predecessor 1 13716 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13715 .coefficient)
      LeftBound13712.bound (LeftBound13712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13716 .coefficient)
      LeftBound13689.bound (LeftBound13689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13689.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13689.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13712.bound, LeftBound13689.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13712.bound, LeftBound13689.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13712.actual selector witness, LeftBound13689.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13717

namespace LeftBound13721
def owner : Owner := ⟨.program ⟨214⟩, ⟨25319⟩⟩
def transferEvent : Nat := 13721
def frameStart : Nat := 13630
def rule : BoundRule := .product (.predecessor 0 13719 .coefficient) (.predecessor 1 13720 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13719 .coefficient)
      LeftBound13717.bound (LeftBound13717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13720 .coefficient)
      LeftAuthority13674.bound (LeftAuthority13674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13717.bound LeftAuthority13674.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13717.bound, LeftAuthority13674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13717.actual selector witness) * (LeftAuthority13674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13721

namespace LeftBound13732
def owner : Owner := ⟨.program ⟨214⟩, ⟨15440⟩⟩
def transferEvent : Nat := 13732
def frameStart : Nat := 13630
def rule : BoundRule := .product (.predecessor 0 13730 .coefficient) (.predecessor 1 13731 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13730 .coefficient)
      LeftAuthority13685.bound (LeftAuthority13685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13685.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13731 .coefficient)
      LeftAuthority13728.bound (LeftAuthority13728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13685.bound LeftAuthority13728.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13685.bound, LeftAuthority13728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority13685.actual selector witness) * (LeftAuthority13728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13732

namespace LeftBound13740
def owner : Owner := ⟨.program ⟨214⟩, ⟨15441⟩⟩
def transferEvent : Nat := 13740
def frameStart : Nat := 13630
def rule : BoundRule := .sum [.predecessor 0 13738 .coefficient, .predecessor 1 13739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13738 .coefficient)
      LeftAuthority13736.bound (LeftAuthority13736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13739 .coefficient)
      LeftBound13732.bound (LeftBound13732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13732.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority13736.bound, LeftBound13732.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13736.bound, LeftBound13732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority13736.actual selector witness, LeftBound13732.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13740

namespace LeftBound13744
def owner : Owner := ⟨.program ⟨214⟩, ⟨25320⟩⟩
def transferEvent : Nat := 13744
def frameStart : Nat := 13630
def rule : BoundRule := .sum [.predecessor 0 13742 .coefficient, .predecessor 1 13743 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13742 .coefficient)
      LeftBound13740.bound (LeftBound13740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13740.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13743 .coefficient)
      LeftBound13721.bound (LeftBound13721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13740.bound, LeftBound13721.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13740.bound, LeftBound13721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13740.actual selector witness, LeftBound13721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13744

namespace LeftBound13757
def owner : Owner := ⟨.program ⟨214⟩, ⟨25318⟩⟩
def transferEvent : Nat := 13757
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13755 .coefficient, .predecessor 1 13756 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13755 .coefficient)
      LeftBound13578.bound (LeftBound13578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13756 .coefficient)
      LeftBound13561.bound (LeftBound13561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13578.bound, LeftBound13561.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13578.bound, LeftBound13561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13578.actual selector witness, LeftBound13561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13757

namespace LeftBound13760
def owner : Owner := ⟨.program ⟨214⟩, ⟨25318⟩⟩
def transferEvent : Nat := 13760
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 13754 .summary, .result 13568 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13754 .summary)
      LeftBound13580.bound (LeftBound13580.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19259⟩⟩) (rawTerms := some (Proof.Events053.exact13754RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13568 .summary)
      LeftBound13563.bound (LeftBound13563.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25317⟩⟩) (rawTerms := some (Proof.Events053.exact13568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13580.bound, LeftBound13563.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13580.bound, LeftBound13563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound13580.actual selector witness, LeftBound13563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13760

namespace LeftBound13764
def owner : Owner := ⟨.program ⟨214⟩, ⟨27052⟩⟩
def transferEvent : Nat := 13764
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13762 .coefficient) (.predecessor 1 13763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13762 .coefficient)
      LeftBound13757.bound (LeftBound13757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13757.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13763 .coefficient)
      LeftAuthority13464.bound (LeftAuthority13464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13757.bound LeftAuthority13464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13757.bound, LeftAuthority13464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13757.actual selector witness) * (LeftAuthority13464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13764

namespace LeftBound13765
def owner : Owner := ⟨.program ⟨214⟩, ⟨27052⟩⟩
def transferEvent : Nat := 13765
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27050⟩⟩]⟩ [⟨.result 13465 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13465 .coefficient)
      LeftAuthority13464.bound (LeftAuthority13464.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27050⟩⟩) (rawTerms := some (Proof.Events052.exact13465RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13464.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13464.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13464.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound13765

namespace LeftBound13766
def owner : Owner := ⟨.program ⟨214⟩, ⟨27052⟩⟩
def transferEvent : Nat := 13766
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 13761 .summary) (.transfer 13765) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13761 .summary)
      LeftBound13760.bound (LeftBound13760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25318⟩⟩) (rawTerms := some (Proof.Events053.exact13761RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound13760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 13765)
      LeftBound13765.bound (LeftBound13765.actual selector witness) := by
  exact .transfer (LeftBound13765.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound13760.bound LeftBound13765.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13760.bound, LeftBound13765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound13760.actual selector witness) * (LeftBound13765.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13766

namespace LeftBound13777
def owner : Owner := ⟨.program ⟨214⟩, ⟨20842⟩⟩
def transferEvent : Nat := 13777
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 13775 .coefficient) (.value (.predecessor 1 13776 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 13775 .coefficient)
      LeftAuthority13773.bound (LeftAuthority13773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events053.exact13774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13773.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13773.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 13776 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority13773.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13773.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13773.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound13777

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
