import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard609

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound89786
def owner : Owner := ⟨.program ⟨214⟩, ⟨18341⟩⟩
def transferEvent : Nat := 89786
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89784 .coefficient, .predecessor 1 89785 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89784 .coefficient)
      LeftBound89782.bound (LeftBound89782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89785 .coefficient)
      LeftAuthority89543.bound (LeftAuthority89543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89543.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89782.bound, LeftAuthority89543.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89782.bound, LeftAuthority89543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89782.actual selector witness, LeftAuthority89543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89786

namespace LeftBound89790
def owner : Owner := ⟨.program ⟨214⟩, ⟨18342⟩⟩
def transferEvent : Nat := 89790
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89788 .coefficient, .predecessor 1 89789 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89788 .coefficient)
      LeftBound89786.bound (LeftBound89786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89789 .coefficient)
      LeftAuthority89520.bound (LeftAuthority89520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89786.bound, LeftAuthority89520.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89786.bound, LeftAuthority89520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89786.actual selector witness, LeftAuthority89520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89790

namespace LeftBound89794
def owner : Owner := ⟨.program ⟨214⟩, ⟨18343⟩⟩
def transferEvent : Nat := 89794
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89792 .coefficient, .predecessor 1 89793 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89792 .coefficient)
      LeftBound89790.bound (LeftBound89790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89790.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89793 .coefficient)
      LeftAuthority89497.bound (LeftAuthority89497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89497.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89790.bound, LeftAuthority89497.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89790.bound, LeftAuthority89497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89790.actual selector witness, LeftAuthority89497.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89794

namespace LeftBound89798
def owner : Owner := ⟨.program ⟨214⟩, ⟨18344⟩⟩
def transferEvent : Nat := 89798
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89796 .coefficient, .predecessor 1 89797 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89796 .coefficient)
      LeftBound89794.bound (LeftBound89794.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89795RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89794.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89797 .coefficient)
      LeftAuthority89474.bound (LeftAuthority89474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89474.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89794.bound, LeftAuthority89474.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89794.bound, LeftAuthority89474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89794.actual selector witness, LeftAuthority89474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89798

namespace LeftBound89802
def owner : Owner := ⟨.program ⟨214⟩, ⟨18345⟩⟩
def transferEvent : Nat := 89802
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89800 .coefficient, .predecessor 1 89801 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89800 .coefficient)
      LeftBound89798.bound (LeftBound89798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89801 .coefficient)
      LeftAuthority89451.bound (LeftAuthority89451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89798.bound, LeftAuthority89451.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89798.bound, LeftAuthority89451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89798.actual selector witness, LeftAuthority89451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89802

namespace LeftBound89806
def owner : Owner := ⟨.program ⟨214⟩, ⟨18346⟩⟩
def transferEvent : Nat := 89806
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89804 .coefficient, .predecessor 1 89805 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89804 .coefficient)
      LeftBound89802.bound (LeftBound89802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89805 .coefficient)
      LeftAuthority89428.bound (LeftAuthority89428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89428.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89802.bound, LeftAuthority89428.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89802.bound, LeftAuthority89428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89802.actual selector witness, LeftAuthority89428.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89806

namespace LeftBound89810
def owner : Owner := ⟨.program ⟨214⟩, ⟨18347⟩⟩
def transferEvent : Nat := 89810
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89808 .coefficient, .predecessor 1 89809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89808 .coefficient)
      LeftBound89806.bound (LeftBound89806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89809 .coefficient)
      LeftAuthority89405.bound (LeftAuthority89405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89806.bound, LeftAuthority89405.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89806.bound, LeftAuthority89405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89806.actual selector witness, LeftAuthority89405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89810

namespace LeftBound89814
def owner : Owner := ⟨.program ⟨214⟩, ⟨18348⟩⟩
def transferEvent : Nat := 89814
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89812 .coefficient, .predecessor 1 89813 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89812 .coefficient)
      LeftBound89810.bound (LeftBound89810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89810.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89813 .coefficient)
      LeftAuthority89382.bound (LeftAuthority89382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89382.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89810.bound, LeftAuthority89382.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89810.bound, LeftAuthority89382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89810.actual selector witness, LeftAuthority89382.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89814

namespace LeftBound89818
def owner : Owner := ⟨.program ⟨214⟩, ⟨18349⟩⟩
def transferEvent : Nat := 89818
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89816 .coefficient, .predecessor 1 89817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89816 .coefficient)
      LeftBound89814.bound (LeftBound89814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89817 .coefficient)
      LeftAuthority89359.bound (LeftAuthority89359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events349.exact89360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89359.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89814.bound, LeftAuthority89359.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89814.bound, LeftAuthority89359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89814.actual selector witness, LeftAuthority89359.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89818

namespace LeftBound89821
def owner : Owner := ⟨.program ⟨214⟩, ⟨18350⟩⟩
def transferEvent : Nat := 89821
def frameStart : Nat := 89317
def rule : BoundRule := .identity (.predecessor 0 89820 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89820 .coefficient)
      LeftBound89818.bound (LeftBound89818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89818.derived selector witness)

def rawBound : CoeffClass := LeftBound89818.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound89818.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound89821

namespace LeftBound89838
def owner : Owner := ⟨.program ⟨214⟩, ⟨18647⟩⟩
def transferEvent : Nat := 89838
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89836 .coefficient, .predecessor 1 89837 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89836 .coefficient)
      LeftBound89821.bound (LeftBound89821.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound89821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89837 .coefficient)
      LeftAuthority89834.bound (LeftAuthority89834.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority89834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89821.bound, LeftAuthority89834.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89821.bound, LeftAuthority89834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89821.actual selector witness, LeftAuthority89834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89838

namespace LeftBound89841
def owner : Owner := ⟨.program ⟨214⟩, ⟨18648⟩⟩
def transferEvent : Nat := 89841
def frameStart : Nat := 89317
def rule : BoundRule := .identity (.predecessor 0 89840 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89840 .coefficient)
      LeftBound89838.bound (LeftBound89838.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound89838.derived selector witness)

def rawBound : CoeffClass := LeftBound89838.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound89838.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound89841

namespace LeftBound89847
def owner : Owner := ⟨.program ⟨214⟩, ⟨18649⟩⟩
def transferEvent : Nat := 89847
def frameStart : Nat := 89317
def rule : BoundRule := .product (.predecessor 0 89845 .coefficient) (.predecessor 1 89846 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89845 .coefficient)
      LeftAuthority89843.bound (LeftAuthority89843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89843.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89846 .coefficient)
      LeftBound89841.bound (LeftBound89841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events350.exact89842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority89843.bound LeftBound89841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority89843.bound, LeftBound89841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority89843.actual selector witness) * (LeftBound89841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound89847

namespace LeftBound89923
def owner : Owner := ⟨.program ⟨214⟩, ⟨6795⟩⟩
def transferEvent : Nat := 89923
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89921 .coefficient, .predecessor 1 89922 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89921 .coefficient)
      LeftAuthority89919.bound (LeftAuthority89919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89922 .coefficient)
      LeftAuthority89916.bound (LeftAuthority89916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority89919.bound, LeftAuthority89916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority89919.bound, LeftAuthority89916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority89919.actual selector witness, LeftAuthority89916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89923

namespace LeftBound89927
def owner : Owner := ⟨.program ⟨214⟩, ⟨6796⟩⟩
def transferEvent : Nat := 89927
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89925 .coefficient, .predecessor 1 89926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89925 .coefficient)
      LeftBound89923.bound (LeftBound89923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89926 .coefficient)
      LeftAuthority89913.bound (LeftAuthority89913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89923.bound, LeftAuthority89913.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89923.bound, LeftAuthority89913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89923.actual selector witness, LeftAuthority89913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89927

namespace LeftBound89931
def owner : Owner := ⟨.program ⟨214⟩, ⟨6797⟩⟩
def transferEvent : Nat := 89931
def frameStart : Nat := 89317
def rule : BoundRule := .sum [.predecessor 0 89929 .coefficient, .predecessor 1 89930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 89929 .coefficient)
      LeftBound89927.bound (LeftBound89927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound89927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound89927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 89930 .coefficient)
      LeftAuthority89910.bound (LeftAuthority89910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events351.exact89911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority89910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority89910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound89927.bound, LeftAuthority89910.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound89927.bound, LeftAuthority89910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound89927.actual selector witness, LeftAuthority89910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound89931

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
