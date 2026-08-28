import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard032
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard642

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94645
def owner : Owner := ⟨.program ⟨214⟩, ⟨22832⟩⟩
def transferEvent : Nat := 94645
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22829⟩⟩]⟩ [⟨.result 94637 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94637 .coefficient)
      LeftAuthority94636.bound (LeftAuthority94636.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22829⟩⟩) (rawTerms := some (Proof.Events369.exact94637RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94636.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94636.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94636.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94636.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94636.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94645

namespace LeftBound94646
def owner : Owner := ⟨.program ⟨214⟩, ⟨22832⟩⟩
def transferEvent : Nat := 94646
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 94645) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94645)
      LeftBound94645.bound (LeftBound94645.actual selector witness) := by
  exact .transfer (LeftBound94645.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound94645.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound94645.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound94645.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94646

namespace LeftBound94717
def owner : Owner := ⟨.program ⟨214⟩, ⟨17002⟩⟩
def transferEvent : Nat := 94717
def frameStart : Nat := 94690
def rule : BoundRule := .identity (.predecessor 0 94716 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94716 .coefficient)
      LeftAuthority94714.bound (LeftAuthority94714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94714.derived selector witness)

def rawBound : CoeffClass := LeftAuthority94714.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority94714.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94717

namespace LeftBound94734
def owner : Owner := ⟨.program ⟨214⟩, ⟨17043⟩⟩
def transferEvent : Nat := 94734
def frameStart : Nat := 94690
def rule : BoundRule := .sum [.predecessor 0 94732 .coefficient, .predecessor 1 94733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94732 .coefficient)
      LeftBound94717.bound (LeftBound94717.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound94717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94733 .coefficient)
      LeftAuthority94730.bound (LeftAuthority94730.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority94730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94717.bound, LeftAuthority94730.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94717.bound, LeftAuthority94730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94717.actual selector witness, LeftAuthority94730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94734

namespace LeftBound94737
def owner : Owner := ⟨.program ⟨214⟩, ⟨17044⟩⟩
def transferEvent : Nat := 94737
def frameStart : Nat := 94690
def rule : BoundRule := .identity (.predecessor 0 94736 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94736 .coefficient)
      LeftBound94734.bound (LeftBound94734.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound94734.derived selector witness)

def rawBound : CoeffClass := LeftBound94734.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound94734.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94737

namespace LeftBound94743
def owner : Owner := ⟨.program ⟨214⟩, ⟨17045⟩⟩
def transferEvent : Nat := 94743
def frameStart : Nat := 94690
def rule : BoundRule := .product (.predecessor 0 94741 .coefficient) (.predecessor 1 94742 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94741 .coefficient)
      LeftAuthority94739.bound (LeftAuthority94739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94742 .coefficient)
      LeftBound94737.bound (LeftBound94737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94737.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority94739.bound LeftBound94737.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94739.bound, LeftBound94737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority94739.actual selector witness) * (LeftBound94737.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94743

namespace LeftBound94751
def owner : Owner := ⟨.program ⟨214⟩, ⟨17046⟩⟩
def transferEvent : Nat := 94751
def frameStart : Nat := 94690
def rule : BoundRule := .sum [.predecessor 0 94749 .coefficient, .predecessor 1 94750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94749 .coefficient)
      LeftAuthority94747.bound (LeftAuthority94747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94747.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94750 .coefficient)
      LeftBound94743.bound (LeftBound94743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94743.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority94747.bound, LeftBound94743.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94747.bound, LeftBound94743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority94747.actual selector witness, LeftBound94743.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94751

namespace LeftBound94755
def owner : Owner := ⟨.program ⟨214⟩, ⟨30062⟩⟩
def transferEvent : Nat := 94755
def frameStart : Nat := 94690
def rule : BoundRule := .product (.predecessor 0 94753 .coefficient) (.predecessor 1 94754 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94753 .coefficient)
      LeftBound94751.bound (LeftBound94751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94754 .coefficient)
      LeftAuthority94728.bound (LeftAuthority94728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94728.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94751.bound LeftAuthority94728.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94751.bound, LeftAuthority94728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94751.actual selector witness) * (LeftAuthority94728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94755

namespace LeftBound94766
def owner : Owner := ⟨.program ⟨214⟩, ⟨18164⟩⟩
def transferEvent : Nat := 94766
def frameStart : Nat := 94690
def rule : BoundRule := .product (.predecessor 0 94764 .coefficient) (.predecessor 1 94765 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94764 .coefficient)
      LeftAuthority94739.bound (LeftAuthority94739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94739.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94765 .coefficient)
      LeftAuthority94762.bound (LeftAuthority94762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority94739.bound LeftAuthority94762.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94739.bound, LeftAuthority94762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority94739.actual selector witness) * (LeftAuthority94762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94766

namespace LeftBound94774
def owner : Owner := ⟨.program ⟨214⟩, ⟨18165⟩⟩
def transferEvent : Nat := 94774
def frameStart : Nat := 94690
def rule : BoundRule := .sum [.predecessor 0 94772 .coefficient, .predecessor 1 94773 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94772 .coefficient)
      LeftAuthority94770.bound (LeftAuthority94770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94770.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94773 .coefficient)
      LeftBound94766.bound (LeftBound94766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94766.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority94770.bound, LeftBound94766.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94770.bound, LeftBound94766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority94770.actual selector witness, LeftBound94766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94774

namespace LeftBound94778
def owner : Owner := ⟨.program ⟨214⟩, ⟨30069⟩⟩
def transferEvent : Nat := 94778
def frameStart : Nat := 94690
def rule : BoundRule := .sum [.predecessor 0 94776 .coefficient, .predecessor 1 94777 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94776 .coefficient)
      LeftBound94774.bound (LeftBound94774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94774.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94777 .coefficient)
      LeftBound94755.bound (LeftBound94755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94774.bound, LeftBound94755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94774.bound, LeftBound94755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94774.actual selector witness, LeftBound94755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94778

namespace LeftBound94791
def owner : Owner := ⟨.program ⟨214⟩, ⟨30064⟩⟩
def transferEvent : Nat := 94791
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94789 .coefficient, .predecessor 1 94790 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94789 .coefficient)
      LeftBound94644.bound (LeftBound94644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94644.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94644.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94790 .coefficient)
      LeftBound94627.bound (LeftBound94627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94627.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94627.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94644.bound, LeftBound94627.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94644.bound, LeftBound94627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94644.actual selector witness, LeftBound94627.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94791

namespace LeftBound94794
def owner : Owner := ⟨.program ⟨214⟩, ⟨30064⟩⟩
def transferEvent : Nat := 94794
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94788 .summary, .result 94634 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94788 .summary)
      LeftBound94646.bound (LeftBound94646.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22832⟩⟩) (rawTerms := some (Proof.Events370.exact94788RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94646.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94634 .summary)
      LeftBound94629.bound (LeftBound94629.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30063⟩⟩) (rawTerms := some (Proof.Events369.exact94634RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94629.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94646.bound, LeftBound94629.bound]
def bound : CoeffClass := .finite ⟨1292539135285018636288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94646.bound, LeftBound94629.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94646.actual selector witness, LeftBound94629.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94794

namespace LeftBound94818
def owner : Owner := ⟨.program ⟨214⟩, ⟨13133⟩⟩
def transferEvent : Nat := 94818
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 94816 .coefficient) (.predecessor 1 94817 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94816 .coefficient)
      LeftAuthority4588.bound (LeftAuthority4588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events017.exact4589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94817 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4588.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4588.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4588.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94818

namespace LeftBound94823
def owner : Owner := ⟨.program ⟨214⟩, ⟨7126⟩⟩
def transferEvent : Nat := 94823
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94821 .coefficient) (.predecessor 1 94822 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94821 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94822 .coefficient)
      LeftBound6972.bound (LeftBound6972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound6972.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound6972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound6972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94823

namespace LeftBound94828
def owner : Owner := ⟨.program ⟨214⟩, ⟨13134⟩⟩
def transferEvent : Nat := 94828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94826 .coefficient, .predecessor 1 94827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94826 .coefficient)
      LeftBound94823.bound (LeftBound94823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94827 .coefficient)
      LeftBound94818.bound (LeftBound94818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events370.exact94820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94818.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94823.bound, LeftBound94818.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94823.bound, LeftBound94818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94823.actual selector witness, LeftBound94818.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94828

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
