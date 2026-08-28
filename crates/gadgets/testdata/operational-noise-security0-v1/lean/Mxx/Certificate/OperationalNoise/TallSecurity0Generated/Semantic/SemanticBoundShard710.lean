import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard707
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard709

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound103565
def owner : Owner := ⟨.program ⟨214⟩, ⟨6806⟩⟩
def transferEvent : Nat := 103565
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103563 .coefficient, .predecessor 1 103564 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103563 .coefficient)
      LeftBound103561.bound (LeftBound103561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103561.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103564 .coefficient)
      LeftAuthority103481.bound (LeftAuthority103481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103561.bound, LeftAuthority103481.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103561.bound, LeftAuthority103481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103561.actual selector witness, LeftAuthority103481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103565

namespace LeftBound103569
def owner : Owner := ⟨.program ⟨214⟩, ⟨6807⟩⟩
def transferEvent : Nat := 103569
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103567 .coefficient, .predecessor 1 103568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103567 .coefficient)
      LeftBound103565.bound (LeftBound103565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103565.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103568 .coefficient)
      LeftAuthority103478.bound (LeftAuthority103478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103565.bound, LeftAuthority103478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103565.bound, LeftAuthority103478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103565.actual selector witness, LeftAuthority103478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103569

namespace LeftBound103573
def owner : Owner := ⟨.program ⟨214⟩, ⟨6808⟩⟩
def transferEvent : Nat := 103573
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103571 .coefficient, .predecessor 1 103572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103571 .coefficient)
      LeftBound103569.bound (LeftBound103569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103572 .coefficient)
      LeftAuthority103475.bound (LeftAuthority103475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103475.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103569.bound, LeftAuthority103475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103569.bound, LeftAuthority103475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103569.actual selector witness, LeftAuthority103475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103573

namespace LeftBound103577
def owner : Owner := ⟨.program ⟨214⟩, ⟨6809⟩⟩
def transferEvent : Nat := 103577
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103575 .coefficient, .predecessor 1 103576 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103575 .coefficient)
      LeftBound103573.bound (LeftBound103573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103576 .coefficient)
      LeftAuthority103472.bound (LeftAuthority103472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103472.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103573.bound, LeftAuthority103472.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103573.bound, LeftAuthority103472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103573.actual selector witness, LeftAuthority103472.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103577

namespace LeftBound103581
def owner : Owner := ⟨.program ⟨214⟩, ⟨6810⟩⟩
def transferEvent : Nat := 103581
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103579 .coefficient, .predecessor 1 103580 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103579 .coefficient)
      LeftBound103577.bound (LeftBound103577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103580 .coefficient)
      LeftAuthority103469.bound (LeftAuthority103469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103469.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103577.bound, LeftAuthority103469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103577.bound, LeftAuthority103469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103577.actual selector witness, LeftAuthority103469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103581

namespace LeftBound103585
def owner : Owner := ⟨.program ⟨214⟩, ⟨6811⟩⟩
def transferEvent : Nat := 103585
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103583 .coefficient, .predecessor 1 103584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103583 .coefficient)
      LeftBound103581.bound (LeftBound103581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103584 .coefficient)
      LeftAuthority103466.bound (LeftAuthority103466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103466.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103581.bound, LeftAuthority103466.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103581.bound, LeftAuthority103466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103581.actual selector witness, LeftAuthority103466.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103585

namespace LeftBound103589
def owner : Owner := ⟨.program ⟨214⟩, ⟨18642⟩⟩
def transferEvent : Nat := 103589
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103587 .coefficient, .predecessor 1 103588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103587 .coefficient)
      LeftBound103585.bound (LeftBound103585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103588 .coefficient)
      LeftBound103445.bound (LeftBound103445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103585.bound, LeftBound103445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103585.bound, LeftBound103445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103585.actual selector witness, LeftBound103445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103589

namespace LeftBound103593
def owner : Owner := ⟨.program ⟨214⟩, ⟨18675⟩⟩
def transferEvent : Nat := 103593
def frameStart : Nat := 102927
def rule : BoundRule := .product (.predecessor 0 103591 .coefficient) (.predecessor 1 103592 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103591 .coefficient)
      LeftBound103589.bound (LeftBound103589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103589.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103592 .coefficient)
      LeftAuthority103430.bound (LeftAuthority103430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103430.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103430.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound103589.bound LeftAuthority103430.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103589.bound, LeftAuthority103430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound103589.actual selector witness) * (LeftAuthority103430.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103593

namespace LeftBound103672
def owner : Owner := ⟨.program ⟨214⟩, ⟨18487⟩⟩
def transferEvent : Nat := 103672
def frameStart : Nat := 102927
def rule : BoundRule := .product (.predecessor 0 103670 .coefficient) (.predecessor 1 103671 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103670 .coefficient)
      LeftAuthority103441.bound (LeftAuthority103441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103671 .coefficient)
      LeftAuthority103668.bound (LeftAuthority103668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103668.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103668.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority103441.bound LeftAuthority103668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103441.bound, LeftAuthority103668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority103441.actual selector witness) * (LeftAuthority103668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103672

namespace LeftBound103680
def owner : Owner := ⟨.program ⟨214⟩, ⟨18488⟩⟩
def transferEvent : Nat := 103680
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103678 .coefficient, .predecessor 1 103679 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103678 .coefficient)
      LeftAuthority103676.bound (LeftAuthority103676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority103676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority103676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103679 .coefficient)
      LeftBound103672.bound (LeftBound103672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103672.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103672.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority103676.bound, LeftBound103672.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority103676.bound, LeftBound103672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority103676.actual selector witness, LeftBound103672.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103680

namespace LeftBound103684
def owner : Owner := ⟨.program ⟨214⟩, ⟨18676⟩⟩
def transferEvent : Nat := 103684
def frameStart : Nat := 102927
def rule : BoundRule := .sum [.predecessor 0 103682 .coefficient, .predecessor 1 103683 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103682 .coefficient)
      LeftBound103680.bound (LeftBound103680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103680.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103683 .coefficient)
      LeftBound103593.bound (LeftBound103593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103593.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103680.bound, LeftBound103593.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103680.bound, LeftBound103593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103680.actual selector witness, LeftBound103593.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103684

namespace LeftBound103731
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def transferEvent : Nat := 103731
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 103729 .coefficient, .predecessor 1 103730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103729 .coefficient)
      LeftBound102346.bound (LeftBound102346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103730 .coefficient)
      LeftBound102261.bound (LeftBound102261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events399.exact102336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102261.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102346.bound, LeftBound102261.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102346.bound, LeftBound102261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102346.actual selector witness, LeftBound102261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103731

namespace LeftBound103768
def owner : Owner := ⟨.program ⟨214⟩, ⟨30067⟩⟩
def transferEvent : Nat := 103768
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 103728 .summary, .result 102336 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103728 .summary)
      LeftBound102348.bound (LeftBound102348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18551⟩⟩) (rawTerms := some (Proof.Events405.exact103728RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102336 .summary)
      LeftBound102263.bound (LeftBound102263.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30066⟩⟩) (rawTerms := some (Proof.Events399.exact102336RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound102348.bound, LeftBound102263.bound]
def bound : CoeffClass := .finite ⟨85361036953731455419885957120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102348.bound, LeftBound102263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound102348.actual selector witness, LeftBound102263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound103768

namespace LeftBound103772
def owner : Owner := ⟨.program ⟨214⟩, ⟨30068⟩⟩
def transferEvent : Nat := 103772
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 103770 .coefficient) (.predecessor 1 103771 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 103770 .coefficient)
      LeftBound103731.bound (LeftBound103731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 103771 .coefficient)
      LeftBound5498.bound (LeftBound5498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound103731.bound LeftBound5498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103731.bound, LeftBound5498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound103731.actual selector witness) * (LeftBound5498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103772

namespace LeftBound103773
def owner : Owner := ⟨.program ⟨214⟩, ⟨30068⟩⟩
def transferEvent : Nat := 103773
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩ [⟨.result 5495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5495 .coefficient)
      LeftAuthority5494.bound (LeftAuthority5494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6651⟩⟩) (rawTerms := some (Proof.Events021.exact5495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5494.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound103773

namespace LeftBound103774
def owner : Owner := ⟨.program ⟨214⟩, ⟨30068⟩⟩
def transferEvent : Nat := 103774
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 103769 .summary) (.transfer 103773) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103769 .summary)
      LeftBound103768.bound (LeftBound103768.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30067⟩⟩) (rawTerms := some (Proof.Events405.exact103769RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 103773)
      LeftBound103773.bound (LeftBound103773.actual selector witness) := by
  exact .transfer (LeftBound103773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound103768.bound LeftBound103773.bound
def bound : CoeffClass := .finite ⟨313276371396785701094268180805713920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103768.bound, LeftBound103773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound103768.actual selector witness) * (LeftBound103773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound103774

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
