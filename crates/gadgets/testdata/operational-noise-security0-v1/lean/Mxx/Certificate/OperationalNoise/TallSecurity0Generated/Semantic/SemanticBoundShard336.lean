import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard009
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50552
def owner : Owner := ⟨.program ⟨214⟩, ⟨7756⟩⟩
def transferEvent : Nat := 50552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50550 .coefficient, .predecessor 1 50551 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50550 .coefficient)
      LeftBound50548.bound (LeftBound50548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50551 .coefficient)
      LeftAuthority50502.bound (LeftAuthority50502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50502.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50548.bound, LeftAuthority50502.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50548.bound, LeftAuthority50502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50548.actual selector witness, LeftAuthority50502.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50552

namespace LeftBound50553
def owner : Owner := ⟨.program ⟨214⟩, ⟨7756⟩⟩
def transferEvent : Nat := 50553
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨69⟩⟩]⟩ [⟨.result 50503 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50503 .coefficient)
      LeftAuthority50502.bound (LeftAuthority50502.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨69⟩⟩) (rawTerms := some (Proof.Events197.exact50503RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50502.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50502.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50502.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50553

namespace LeftBound50558
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50558
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50556 .coefficient) (.predecessor 1 50557 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50556 .coefficient)
      LeftBound50552.bound (LeftBound50552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50557 .coefficient)
      LeftBound3047.bound (LeftBound3047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50552.bound LeftBound3047.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50552.bound, LeftBound3047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50552.actual selector witness) * (LeftBound3047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50558

namespace LeftBound50559
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50559
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18499⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 2823 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6410⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2823 .coefficient)
      LeftAuthority2822.bound (LeftAuthority2822.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18499⟩⟩) (rawTerms := some (Proof.Events011.exact2823RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2822.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority2822.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority2822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority2822.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50559

namespace LeftBound50560
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50560
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18128⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 2831 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6435⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2831 .coefficient)
      LeftAuthority2830.bound (LeftAuthority2830.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18128⟩⟩) (rawTerms := some (Proof.Events011.exact2831RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2830.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority2830.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority2830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority2830.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50560

namespace LeftBound50561
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50561
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50559, .transfer 50560]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50559)
      LeftBound50559.bound (LeftBound50559.actual selector witness) := by
  exact .transfer (LeftBound50559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50560)
      LeftBound50560.bound (LeftBound50560.actual selector witness) := by
  exact .transfer (LeftBound50560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50559.bound, LeftBound50560.bound]
def bound : CoeffClass := .finite ⟨4453112970957156472086120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50559.bound, LeftBound50560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50559.actual selector witness, LeftBound50560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50561

namespace LeftBound50562
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50562
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16931⟩⟩], []⟩ [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 2839 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 553 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6437⟩⟩) (rawTerms := some (Proof.Events002.exact553RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2839 .coefficient)
      LeftAuthority2838.bound (LeftAuthority2838.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨16931⟩⟩) (rawTerms := some (Proof.Events011.exact2839RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2838.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority552.bound [LeftAuthority2838.bound]
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority552.bound, LeftAuthority2838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority552.actual selector witness) * ([LeftAuthority2838.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50562

namespace LeftBound50563
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50563
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50561, .transfer 50562]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50561)
      LeftBound50561.bound (LeftBound50561.actual selector witness) := by
  exact .transfer (LeftBound50561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50562)
      LeftBound50562.bound (LeftBound50562.actual selector witness) := by
  exact .transfer (LeftBound50562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50561.bound, LeftBound50562.bound]
def bound : CoeffClass := .finite ⟨4683713856341753228595600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50561.bound, LeftBound50562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50561.actual selector witness, LeftBound50562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50563

namespace LeftBound50564
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50564
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17498⟩⟩], []⟩ [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 2847 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 563 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6449⟩⟩) (rawTerms := some (Proof.Events002.exact563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2847 .coefficient)
      LeftAuthority2846.bound (LeftAuthority2846.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17498⟩⟩) (rawTerms := some (Proof.Events011.exact2847RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2846.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority562.bound [LeftAuthority2846.bound]
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority562.bound, LeftAuthority2846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority562.actual selector witness) * ([LeftAuthority2846.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50564

namespace LeftBound50565
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50565
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50563, .transfer 50564]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50563)
      LeftBound50563.bound (LeftBound50563.actual selector witness) := by
  exact .transfer (LeftBound50563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50564)
      LeftBound50564.bound (LeftBound50564.actual selector witness) := by
  exact .transfer (LeftBound50564.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50563.bound, LeftBound50564.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50563.bound, LeftBound50564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50563.actual selector witness, LeftBound50564.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50565

namespace LeftBound50566
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50566
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17722⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 2855 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6459⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2855 .coefficient)
      LeftAuthority2854.bound (LeftAuthority2854.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17722⟩⟩) (rawTerms := some (Proof.Events011.exact2855RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2854.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority2854.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority2854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority2854.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50566

namespace LeftBound50567
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50567
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50565, .transfer 50566]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50565)
      LeftBound50565.bound (LeftBound50565.actual selector witness) := by
  exact .transfer (LeftBound50565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50566)
      LeftBound50566.bound (LeftBound50566.actual selector witness) := by
  exact .transfer (LeftBound50566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50565.bound, LeftBound50566.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50565.bound, LeftBound50566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50565.actual selector witness, LeftBound50566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50567

namespace LeftBound50568
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50568
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17953⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 2863 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6467⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2863 .coefficient)
      LeftAuthority2862.bound (LeftAuthority2862.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17953⟩⟩) (rawTerms := some (Proof.Events011.exact2863RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2862.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority2862.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority2862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority2862.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50568

namespace LeftBound50569
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50569
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50567, .transfer 50568]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50567)
      LeftBound50567.bound (LeftBound50567.actual selector witness) := by
  exact .transfer (LeftBound50567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50568)
      LeftBound50568.bound (LeftBound50568.actual selector witness) := by
  exact .transfer (LeftBound50568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50567.bound, LeftBound50568.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50567.bound, LeftBound50568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50567.actual selector witness, LeftBound50568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50569

namespace LeftBound50570
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50570
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17554⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 2871 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6473⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2871 .coefficient)
      LeftAuthority2870.bound (LeftAuthority2870.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17554⟩⟩) (rawTerms := some (Proof.Events011.exact2871RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2870.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority2870.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority2870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority2870.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound50570

namespace LeftBound50571
def owner : Owner := ⟨.program ⟨214⟩, ⟨18861⟩⟩
def transferEvent : Nat := 50571
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 50569, .transfer 50570]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50569)
      LeftBound50569.bound (LeftBound50569.actual selector witness) := by
  exact .transfer (LeftBound50569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50570)
      LeftBound50570.bound (LeftBound50570.actual selector witness) := by
  exact .transfer (LeftBound50570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50569.bound, LeftBound50570.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50569.bound, LeftBound50570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50569.actual selector witness, LeftBound50570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50571

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
