import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard010
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard011

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound3779
def owner : Owner := ⟨.program ⟨214⟩, ⟨18823⟩⟩
def transferEvent : Nat := 3779
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3777 .coefficient, .predecessor 1 3778 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3777 .coefficient)
      LeftBound3775.bound (LeftBound3775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3778 .coefficient)
      LeftBound3606.bound (LeftBound3606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3775.bound, LeftBound3606.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3775.bound, LeftBound3606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound3775.actual selector witness, LeftBound3606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3779

namespace LeftBound3783
def owner : Owner := ⟨.program ⟨214⟩, ⟨18824⟩⟩
def transferEvent : Nat := 3783
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3781 .coefficient, .predecessor 1 3782 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3781 .coefficient)
      LeftBound3779.bound (LeftBound3779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3779.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3782 .coefficient)
      LeftBound3598.bound (LeftBound3598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3779.bound, LeftBound3598.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3779.bound, LeftBound3598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound3779.actual selector witness, LeftBound3598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3783

namespace LeftBound3787
def owner : Owner := ⟨.program ⟨214⟩, ⟨18825⟩⟩
def transferEvent : Nat := 3787
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3785 .coefficient, .predecessor 1 3786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3785 .coefficient)
      LeftBound3783.bound (LeftBound3783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3783.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3786 .coefficient)
      LeftBound3590.bound (LeftBound3590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3783.bound, LeftBound3590.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3783.bound, LeftBound3590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound3783.actual selector witness, LeftBound3590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3787

namespace LeftBound3791
def owner : Owner := ⟨.program ⟨214⟩, ⟨18826⟩⟩
def transferEvent : Nat := 3791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3789 .coefficient, .predecessor 1 3790 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3789 .coefficient)
      LeftBound3787.bound (LeftBound3787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3790 .coefficient)
      LeftBound3582.bound (LeftBound3582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3787.bound, LeftBound3582.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3787.bound, LeftBound3582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound3787.actual selector witness, LeftBound3582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3791

namespace LeftBound3795
def owner : Owner := ⟨.program ⟨214⟩, ⟨18828⟩⟩
def transferEvent : Nat := 3795
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3793 .coefficient, .predecessor 1 3794 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3793 .coefficient)
      LeftBound3791.bound (LeftBound3791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3791.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3791.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3794 .coefficient)
      LeftBound3574.bound (LeftBound3574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3791.bound, LeftBound3574.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3791.bound, LeftBound3574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound3791.actual selector witness, LeftBound3574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3795

namespace LeftBound3799
def owner : Owner := ⟨.program ⟨214⟩, ⟨18829⟩⟩
def transferEvent : Nat := 3799
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 3797 .coefficient) (.predecessor 1 3798 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 3797 .coefficient)
      LeftBound3795.bound (LeftBound3795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 3798 .coefficient)
      LeftAuthority3072.bound (LeftAuthority3072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3072.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound3795.bound LeftAuthority3072.bound
def bound : CoeffClass := .finite ⟨2427741588940687025667331754114644356019101566559943145012217930194672472934387351200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3795.bound, LeftAuthority3072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound3795.actual selector witness) * (LeftAuthority3072.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound3799

namespace LeftBound4316
def owner : Owner := ⟨.program ⟨214⟩, ⟨18496⟩⟩
def transferEvent : Nat := 4316
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4314 .coefficient) (.predecessor 1 4315 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4314 .coefficient)
      LeftAuthority4312.bound (LeftAuthority4312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4315 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4312.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4312.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4312.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4316

namespace LeftBound4324
def owner : Owner := ⟨.program ⟨214⟩, ⟨18125⟩⟩
def transferEvent : Nat := 4324
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4322 .coefficient) (.predecessor 1 4323 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4322 .coefficient)
      LeftAuthority4320.bound (LeftAuthority4320.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4323 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4320.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4320.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4320.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4324

namespace LeftBound4332
def owner : Owner := ⟨.program ⟨214⟩, ⟨16928⟩⟩
def transferEvent : Nat := 4332
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4330 .coefficient) (.predecessor 1 4331 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4330 .coefficient)
      LeftAuthority4328.bound (LeftAuthority4328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4328.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4331 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4328.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4328.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4328.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4332

namespace LeftBound4340
def owner : Owner := ⟨.program ⟨214⟩, ⟨17495⟩⟩
def transferEvent : Nat := 4340
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4338 .coefficient) (.predecessor 1 4339 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4338 .coefficient)
      LeftAuthority4336.bound (LeftAuthority4336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4336.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4339 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4336.bound LeftAuthority562.bound
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4336.bound, LeftAuthority562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4336.actual selector witness) * (LeftAuthority562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4340

namespace LeftBound4348
def owner : Owner := ⟨.program ⟨214⟩, ⟨17719⟩⟩
def transferEvent : Nat := 4348
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4346 .coefficient) (.predecessor 1 4347 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4346 .coefficient)
      LeftAuthority4344.bound (LeftAuthority4344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4344.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4347 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4344.bound LeftAuthority572.bound
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4344.bound, LeftAuthority572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4344.actual selector witness) * (LeftAuthority572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4348

namespace LeftBound4356
def owner : Owner := ⟨.program ⟨214⟩, ⟨17950⟩⟩
def transferEvent : Nat := 4356
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4354 .coefficient) (.predecessor 1 4355 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4354 .coefficient)
      LeftAuthority4352.bound (LeftAuthority4352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4352.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4355 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4352.bound LeftAuthority582.bound
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4352.bound, LeftAuthority582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4352.actual selector witness) * (LeftAuthority582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4356

namespace LeftBound4364
def owner : Owner := ⟨.program ⟨214⟩, ⟨17551⟩⟩
def transferEvent : Nat := 4364
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4362 .coefficient) (.predecessor 1 4363 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4362 .coefficient)
      LeftAuthority4360.bound (LeftAuthority4360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4360.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4363 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4360.bound LeftAuthority592.bound
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4360.bound, LeftAuthority592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4360.actual selector witness) * (LeftAuthority592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4364

namespace LeftBound4372
def owner : Owner := ⟨.program ⟨214⟩, ⟨18833⟩⟩
def transferEvent : Nat := 4372
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4370 .coefficient) (.predecessor 1 4371 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4370 .coefficient)
      LeftAuthority4368.bound (LeftAuthority4368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4368.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4371 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4368.bound LeftAuthority602.bound
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4368.bound, LeftAuthority602.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4368.actual selector witness) * (LeftAuthority602.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4372

namespace LeftBound4380
def owner : Owner := ⟨.program ⟨214⟩, ⟨17607⟩⟩
def transferEvent : Nat := 4380
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4378 .coefficient) (.predecessor 1 4379 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4378 .coefficient)
      LeftAuthority4376.bound (LeftAuthority4376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4379 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4376.bound LeftAuthority612.bound
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4376.bound, LeftAuthority612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4376.actual selector witness) * (LeftAuthority612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4380

namespace LeftBound4388
def owner : Owner := ⟨.program ⟨214⟩, ⟨17663⟩⟩
def transferEvent : Nat := 4388
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 4386 .coefficient) (.predecessor 1 4387 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 4386 .coefficient)
      LeftAuthority4384.bound (LeftAuthority4384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 4387 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority4384.bound LeftAuthority622.bound
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4384.bound, LeftAuthority622.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority4384.actual selector witness) * (LeftAuthority622.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound4388

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
