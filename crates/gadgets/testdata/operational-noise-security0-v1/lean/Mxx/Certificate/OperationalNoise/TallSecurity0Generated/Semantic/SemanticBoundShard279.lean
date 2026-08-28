import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard278

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41547
def owner : Owner := ⟨.program ⟨214⟩, ⟨14106⟩⟩
def transferEvent : Nat := 41547
def frameStart : Nat := 41494
def rule : BoundRule := .identity (.predecessor 0 41546 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41546 .coefficient)
      LeftBound41544.bound (LeftBound41544.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound41544.derived selector witness)

def rawBound : CoeffClass := LeftBound41544.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound41544.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41547

namespace LeftBound41553
def owner : Owner := ⟨.program ⟨214⟩, ⟨14107⟩⟩
def transferEvent : Nat := 41553
def frameStart : Nat := 41494
def rule : BoundRule := .product (.predecessor 0 41551 .coefficient) (.predecessor 1 41552 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41551 .coefficient)
      LeftAuthority41549.bound (LeftAuthority41549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41552 .coefficient)
      LeftBound41547.bound (LeftBound41547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41547.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority41549.bound LeftBound41547.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41549.bound, LeftBound41547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority41549.actual selector witness) * (LeftBound41547.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41553

namespace LeftBound41569
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 41569
def frameStart : Nat := 41494
def rule : BoundRule := .scale (.predecessor 0 41567 .coefficient) (.value (.predecessor 1 41568 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41567 .coefficient)
      LeftAuthority41565.bound (LeftAuthority41565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41568 .coefficient)
      LeftAuthority41556.bound (LeftAuthority41556.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority41556.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41565.bound LeftAuthority41556.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41565.bound, LeftAuthority41556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41565.actual selector witness) * (LeftAuthority41556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41569

namespace LeftBound41572
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 41572
def frameStart : Nat := 41494
def rule : BoundRule := .identity (.predecessor 0 41571 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41571 .coefficient)
      LeftAuthority41559.bound (LeftAuthority41559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41559.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41559.derived selector witness)

def rawBound : CoeffClass := LeftAuthority41559.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority41559.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound41572

namespace LeftBound41576
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 41576
def frameStart : Nat := 41494
def rule : BoundRule := .product (.predecessor 0 41574 .coefficient) (.predecessor 1 41575 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41574 .coefficient)
      LeftBound41572.bound (LeftBound41572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41572.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41575 .coefficient)
      LeftBound41569.bound (LeftBound41569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41569.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41572.bound LeftBound41569.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41572.bound, LeftBound41569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41572.actual selector witness) * (LeftBound41569.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41576

namespace LeftBound41581
def owner : Owner := ⟨.program ⟨214⟩, ⟨14108⟩⟩
def transferEvent : Nat := 41581
def frameStart : Nat := 41494
def rule : BoundRule := .sum [.predecessor 0 41579 .coefficient, .predecessor 1 41580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41579 .coefficient)
      LeftBound41576.bound (LeftBound41576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41580 .coefficient)
      LeftBound41553.bound (LeftBound41553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41576.bound, LeftBound41553.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41576.bound, LeftBound41553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41576.actual selector witness, LeftBound41553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41581

namespace LeftBound41585
def owner : Owner := ⟨.program ⟨214⟩, ⟨26002⟩⟩
def transferEvent : Nat := 41585
def frameStart : Nat := 41494
def rule : BoundRule := .product (.predecessor 0 41583 .coefficient) (.predecessor 1 41584 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41583 .coefficient)
      LeftBound41581.bound (LeftBound41581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41584 .coefficient)
      LeftAuthority41538.bound (LeftAuthority41538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41538.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41581.bound LeftAuthority41538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41581.bound, LeftAuthority41538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41581.actual selector witness) * (LeftAuthority41538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41585

namespace LeftBound41596
def owner : Owner := ⟨.program ⟨214⟩, ⟨15831⟩⟩
def transferEvent : Nat := 41596
def frameStart : Nat := 41494
def rule : BoundRule := .product (.predecessor 0 41594 .coefficient) (.predecessor 1 41595 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41594 .coefficient)
      LeftAuthority41549.bound (LeftAuthority41549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41595 .coefficient)
      LeftAuthority41592.bound (LeftAuthority41592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority41549.bound LeftAuthority41592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41549.bound, LeftAuthority41592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority41549.actual selector witness) * (LeftAuthority41592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41596

namespace LeftBound41604
def owner : Owner := ⟨.program ⟨214⟩, ⟨15832⟩⟩
def transferEvent : Nat := 41604
def frameStart : Nat := 41494
def rule : BoundRule := .sum [.predecessor 0 41602 .coefficient, .predecessor 1 41603 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41602 .coefficient)
      LeftAuthority41600.bound (LeftAuthority41600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41600.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41603 .coefficient)
      LeftBound41596.bound (LeftBound41596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority41600.bound, LeftBound41596.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41600.bound, LeftBound41596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority41600.actual selector witness, LeftBound41596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41604

namespace LeftBound41608
def owner : Owner := ⟨.program ⟨214⟩, ⟨26003⟩⟩
def transferEvent : Nat := 41608
def frameStart : Nat := 41494
def rule : BoundRule := .sum [.predecessor 0 41606 .coefficient, .predecessor 1 41607 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41606 .coefficient)
      LeftBound41604.bound (LeftBound41604.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41605RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41604.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41604.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41607 .coefficient)
      LeftBound41585.bound (LeftBound41585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41604.bound, LeftBound41585.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41604.bound, LeftBound41585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41604.actual selector witness, LeftBound41585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41608

namespace LeftBound41621
def owner : Owner := ⟨.program ⟨214⟩, ⟨26001⟩⟩
def transferEvent : Nat := 41621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 41619 .coefficient, .predecessor 1 41620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41619 .coefficient)
      LeftBound41442.bound (LeftBound41442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41620 .coefficient)
      LeftBound41425.bound (LeftBound41425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41442.bound, LeftBound41425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41442.bound, LeftBound41425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41442.actual selector witness, LeftBound41425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41621

namespace LeftBound41624
def owner : Owner := ⟨.program ⟨214⟩, ⟨26001⟩⟩
def transferEvent : Nat := 41624
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 41618 .summary, .result 41432 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41618 .summary)
      LeftBound41444.bound (LeftBound41444.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19467⟩⟩) (rawTerms := some (Proof.Events162.exact41618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41432 .summary)
      LeftBound41427.bound (LeftBound41427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26000⟩⟩) (rawTerms := some (Proof.Events161.exact41432RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound41444.bound, LeftBound41427.bound]
def bound : CoeffClass := .finite ⟨352054612209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41444.bound, LeftBound41427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound41444.actual selector witness, LeftBound41427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound41624

namespace LeftBound41628
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def transferEvent : Nat := 41628
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41626 .coefficient) (.predecessor 1 41627 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41626 .coefficient)
      LeftBound41621.bound (LeftBound41621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41627 .coefficient)
      LeftAuthority41347.bound (LeftAuthority41347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events161.exact41348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41347.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41621.bound LeftAuthority41347.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41621.bound, LeftAuthority41347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41621.actual selector witness) * (LeftAuthority41347.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41628

namespace LeftBound41629
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def transferEvent : Nat := 41629
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27675⟩⟩]⟩ [⟨.result 41348 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41348 .coefficient)
      LeftAuthority41347.bound (LeftAuthority41347.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27675⟩⟩) (rawTerms := some (Proof.Events161.exact41348RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41347.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41347.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41347.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41347.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41629

namespace LeftBound41630
def owner : Owner := ⟨.program ⟨214⟩, ⟨27677⟩⟩
def transferEvent : Nat := 41630
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41625 .summary) (.transfer 41629) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41625 .summary)
      LeftBound41624.bound (LeftBound41624.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26001⟩⟩) (rawTerms := some (Proof.Events162.exact41625RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41629)
      LeftBound41629.bound (LeftBound41629.actual selector witness) := by
  exact .transfer (LeftBound41629.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41624.bound LeftBound41629.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41624.bound, LeftBound41629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41624.actual selector witness) * (LeftBound41629.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41630

namespace LeftBound41641
def owner : Owner := ⟨.program ⟨214⟩, ⟨21266⟩⟩
def transferEvent : Nat := 41641
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 41639 .coefficient) (.value (.predecessor 1 41640 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41639 .coefficient)
      LeftAuthority41637.bound (LeftAuthority41637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events162.exact41638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41640 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41637.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41637.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41637.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41641

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
