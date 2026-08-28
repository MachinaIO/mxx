import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard708

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound103416
def owner : Owner := ⟨.program ⟨214⟩, ⟨18312⟩⟩
def transferEvent : Nat := 103416
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103414 .coefficient, .predecessor 1 103415 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103414 .coefficient)
      LeftBound103412.bound (LeftBound103412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103415 .coefficient)
      LeftAuthority102957.bound (LeftAuthority102957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact102958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority102957.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority102957.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103412.bound, LeftAuthority102957.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103412.bound, LeftAuthority102957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103412.actual selector witness, LeftAuthority102957.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103416

namespace LeftBound103419
def owner : Owner := ⟨.program ⟨214⟩, ⟨18313⟩⟩
def transferEvent : Nat := 103419
def frameStart : Nat := 102927
def rule : BoundRule := .identity (.predecessor 0 103418 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103418 .coefficient)
      LeftBound103416.bound (LeftBound103416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103416.derived selector witness)

def rawBound : CoeffClass := LeftBound103416.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound103416.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound103419

namespace LeftBound103436
def owner : Owner := ⟨.program ⟨214⟩, ⟨18639⟩⟩
def transferEvent : Nat := 103436
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103434 .coefficient, .predecessor 1 103435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103434 .coefficient)
      LeftBound103419.bound (LeftBound103419.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound103419.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103435 .coefficient)
      LeftAuthority103432.bound (LeftAuthority103432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority103432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103419.bound, LeftAuthority103432.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103419.bound, LeftAuthority103432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103419.actual selector witness, LeftAuthority103432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103436

namespace LeftBound103439
def owner : Owner := ⟨.program ⟨214⟩, ⟨18640⟩⟩
def transferEvent : Nat := 103439
def frameStart : Nat := 102927
def rule : BoundRule := .identity (.predecessor 0 103438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103438 .coefficient)
      LeftBound103436.bound (LeftBound103436.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound103436.derived selector witness)

def rawBound : CoeffClass := LeftBound103436.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound103436.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound103439

namespace LeftBound103445
def owner : Owner := ⟨.program ⟨214⟩, ⟨18641⟩⟩
def transferEvent : Nat := 103445
def frameStart : Nat := 102927
def rule : BoundRule := .product (.predecessor 0 103443 .coefficient) (.predecessor 1 103444 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103443 .coefficient)
      LeftAuthority103441.bound (LeftAuthority103441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103444 .coefficient)
      LeftBound103439.bound (LeftBound103439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority103441.bound LeftBound103439.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103441.bound, LeftBound103439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority103441.actual selector witness) * (LeftBound103439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103445

namespace LeftBound103521
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 103521
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103519 .coefficient, .predecessor 1 103520 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103519 .coefficient)
      LeftAuthority103517.bound (LeftAuthority103517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103517.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103520 .coefficient)
      LeftAuthority103514.bound (LeftAuthority103514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103517.bound, LeftAuthority103514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103517.bound, LeftAuthority103514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority103517.actual selector witness, LeftAuthority103514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103521

namespace LeftBound103525
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 103525
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103523 .coefficient, .predecessor 1 103524 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103523 .coefficient)
      LeftBound103521.bound (LeftBound103521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103524 .coefficient)
      LeftAuthority103511.bound (LeftAuthority103511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103511.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103521.bound, LeftAuthority103511.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103521.bound, LeftAuthority103511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103521.actual selector witness, LeftAuthority103511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103525

namespace LeftBound103529
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 103529
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103527 .coefficient, .predecessor 1 103528 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103527 .coefficient)
      LeftBound103525.bound (LeftBound103525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103525.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103525.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103528 .coefficient)
      LeftAuthority103508.bound (LeftAuthority103508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103508.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103525.bound, LeftAuthority103508.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103525.bound, LeftAuthority103508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103525.actual selector witness, LeftAuthority103508.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103529

namespace LeftBound103533
def owner : Owner := ⟨.program ⟨214⟩, ⟨6798⟩⟩
def transferEvent : Nat := 103533
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103531 .coefficient, .predecessor 1 103532 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103531 .coefficient)
      LeftBound103529.bound (LeftBound103529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103529.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103529.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103532 .coefficient)
      LeftAuthority103505.bound (LeftAuthority103505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103529.bound, LeftAuthority103505.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103529.bound, LeftAuthority103505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103529.actual selector witness, LeftAuthority103505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103533

namespace LeftBound103537
def owner : Owner := ⟨.program ⟨214⟩, ⟨6799⟩⟩
def transferEvent : Nat := 103537
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103535 .coefficient, .predecessor 1 103536 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103535 .coefficient)
      LeftBound103533.bound (LeftBound103533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103536 .coefficient)
      LeftAuthority103502.bound (LeftAuthority103502.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103502.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103502.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103533.bound, LeftAuthority103502.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103533.bound, LeftAuthority103502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103533.actual selector witness, LeftAuthority103502.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103537

namespace LeftBound103541
def owner : Owner := ⟨.program ⟨214⟩, ⟨6800⟩⟩
def transferEvent : Nat := 103541
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103539 .coefficient, .predecessor 1 103540 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103539 .coefficient)
      LeftBound103537.bound (LeftBound103537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103540 .coefficient)
      LeftAuthority103499.bound (LeftAuthority103499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103499.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103537.bound, LeftAuthority103499.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103537.bound, LeftAuthority103499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103537.actual selector witness, LeftAuthority103499.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103541

namespace LeftBound103545
def owner : Owner := ⟨.program ⟨214⟩, ⟨6801⟩⟩
def transferEvent : Nat := 103545
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103543 .coefficient, .predecessor 1 103544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103543 .coefficient)
      LeftBound103541.bound (LeftBound103541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103544 .coefficient)
      LeftAuthority103496.bound (LeftAuthority103496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103541.bound, LeftAuthority103496.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103541.bound, LeftAuthority103496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103541.actual selector witness, LeftAuthority103496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103545

namespace LeftBound103549
def owner : Owner := ⟨.program ⟨214⟩, ⟨6802⟩⟩
def transferEvent : Nat := 103549
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103547 .coefficient, .predecessor 1 103548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103547 .coefficient)
      LeftBound103545.bound (LeftBound103545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103548 .coefficient)
      LeftAuthority103493.bound (LeftAuthority103493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103545.bound, LeftAuthority103493.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103545.bound, LeftAuthority103493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103545.actual selector witness, LeftAuthority103493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103549

namespace LeftBound103553
def owner : Owner := ⟨.program ⟨214⟩, ⟨6803⟩⟩
def transferEvent : Nat := 103553
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103551 .coefficient, .predecessor 1 103552 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103551 .coefficient)
      LeftBound103549.bound (LeftBound103549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103552 .coefficient)
      LeftAuthority103490.bound (LeftAuthority103490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103490.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103549.bound, LeftAuthority103490.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103549.bound, LeftAuthority103490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103549.actual selector witness, LeftAuthority103490.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103553

namespace LeftBound103557
def owner : Owner := ⟨.program ⟨214⟩, ⟨6804⟩⟩
def transferEvent : Nat := 103557
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103555 .coefficient, .predecessor 1 103556 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103555 .coefficient)
      LeftBound103553.bound (LeftBound103553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103556 .coefficient)
      LeftAuthority103487.bound (LeftAuthority103487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103487.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103487.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103553.bound, LeftAuthority103487.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103553.bound, LeftAuthority103487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103553.actual selector witness, LeftAuthority103487.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103557

namespace LeftBound103561
def owner : Owner := ⟨.program ⟨214⟩, ⟨6805⟩⟩
def transferEvent : Nat := 103561
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103559 .coefficient, .predecessor 1 103560 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103559 .coefficient)
      LeftBound103557.bound (LeftBound103557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103557.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103557.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103560 .coefficient)
      LeftAuthority103484.bound (LeftAuthority103484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103484.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103557.bound, LeftAuthority103484.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103557.bound, LeftAuthority103484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103557.actual selector witness, LeftAuthority103484.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103561

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
