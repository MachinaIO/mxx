import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard127

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20817
def owner : Owner := ⟨.program ⟨214⟩, ⟨14809⟩⟩
def transferEvent : Nat := 20817
def frameStart : Nat := 20778
def rule : BoundRule := .identity (.predecessor 0 20816 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20816 .coefficient)
      LeftAuthority20814.bound (LeftAuthority20814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20814.derived selector witness)

def rawBound : CoeffClass := LeftAuthority20814.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority20814.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20817

namespace LeftBound20834
def owner : Owner := ⟨.program ⟨214⟩, ⟨14848⟩⟩
def transferEvent : Nat := 20834
def frameStart : Nat := 20778
def rule : BoundRule := .sum [.predecessor 0 20832 .coefficient, .predecessor 1 20833 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20832 .coefficient)
      LeftBound20817.bound (LeftBound20817.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20833 .coefficient)
      LeftAuthority20830.bound (LeftAuthority20830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority20830.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20817.bound, LeftAuthority20830.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20817.bound, LeftAuthority20830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20817.actual selector witness, LeftAuthority20830.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20834

namespace LeftBound20837
def owner : Owner := ⟨.program ⟨214⟩, ⟨14849⟩⟩
def transferEvent : Nat := 20837
def frameStart : Nat := 20778
def rule : BoundRule := .identity (.predecessor 0 20836 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20836 .coefficient)
      LeftBound20834.bound (LeftBound20834.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20834.derived selector witness)

def rawBound : CoeffClass := LeftBound20834.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound20834.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20837

namespace LeftBound20843
def owner : Owner := ⟨.program ⟨214⟩, ⟨14850⟩⟩
def transferEvent : Nat := 20843
def frameStart : Nat := 20778
def rule : BoundRule := .product (.predecessor 0 20841 .coefficient) (.predecessor 1 20842 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20841 .coefficient)
      LeftAuthority20839.bound (LeftAuthority20839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20842 .coefficient)
      LeftBound20837.bound (LeftBound20837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20837.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority20839.bound LeftBound20837.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20839.bound, LeftBound20837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority20839.actual selector witness) * (LeftBound20837.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20843

namespace LeftBound20851
def owner : Owner := ⟨.program ⟨214⟩, ⟨14851⟩⟩
def transferEvent : Nat := 20851
def frameStart : Nat := 20778
def rule : BoundRule := .sum [.predecessor 0 20849 .coefficient, .predecessor 1 20850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20849 .coefficient)
      LeftAuthority20847.bound (LeftAuthority20847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20850 .coefficient)
      LeftBound20843.bound (LeftBound20843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20847.bound, LeftBound20843.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20847.bound, LeftBound20843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20847.actual selector witness, LeftBound20843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20851

namespace LeftBound20855
def owner : Owner := ⟨.program ⟨214⟩, ⟨26400⟩⟩
def transferEvent : Nat := 20855
def frameStart : Nat := 20778
def rule : BoundRule := .product (.predecessor 0 20853 .coefficient) (.predecessor 1 20854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20853 .coefficient)
      LeftBound20851.bound (LeftBound20851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20854 .coefficient)
      LeftAuthority20828.bound (LeftAuthority20828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20851.bound LeftAuthority20828.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20851.bound, LeftAuthority20828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20851.actual selector witness) * (LeftAuthority20828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20855

namespace LeftBound20866
def owner : Owner := ⟨.program ⟨214⟩, ⟨14909⟩⟩
def transferEvent : Nat := 20866
def frameStart : Nat := 20778
def rule : BoundRule := .product (.predecessor 0 20864 .coefficient) (.predecessor 1 20865 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20864 .coefficient)
      LeftAuthority20839.bound (LeftAuthority20839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20865 .coefficient)
      LeftAuthority20862.bound (LeftAuthority20862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority20839.bound LeftAuthority20862.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20839.bound, LeftAuthority20862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority20839.actual selector witness) * (LeftAuthority20862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20866

namespace LeftBound20874
def owner : Owner := ⟨.program ⟨214⟩, ⟨14910⟩⟩
def transferEvent : Nat := 20874
def frameStart : Nat := 20778
def rule : BoundRule := .sum [.predecessor 0 20872 .coefficient, .predecessor 1 20873 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20872 .coefficient)
      LeftAuthority20870.bound (LeftAuthority20870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20873 .coefficient)
      LeftBound20866.bound (LeftBound20866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20870.bound, LeftBound20866.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20870.bound, LeftBound20866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20870.actual selector witness, LeftBound20866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20874

namespace LeftBound20878
def owner : Owner := ⟨.program ⟨214⟩, ⟨26405⟩⟩
def transferEvent : Nat := 20878
def frameStart : Nat := 20778
def rule : BoundRule := .sum [.predecessor 0 20876 .coefficient, .predecessor 1 20877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20876 .coefficient)
      LeftBound20874.bound (LeftBound20874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20877 .coefficient)
      LeftBound20855.bound (LeftBound20855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20874.bound, LeftBound20855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20874.bound, LeftBound20855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20874.actual selector witness, LeftBound20855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20878

namespace LeftBound20891
def owner : Owner := ⟨.program ⟨214⟩, ⟨26402⟩⟩
def transferEvent : Nat := 20891
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20889 .coefficient, .predecessor 1 20890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20889 .coefficient)
      LeftBound20720.bound (LeftBound20720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20890 .coefficient)
      LeftBound20703.bound (LeftBound20703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20720.bound, LeftBound20703.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20720.bound, LeftBound20703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20720.actual selector witness, LeftBound20703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20891

namespace LeftBound20894
def owner : Owner := ⟨.program ⟨214⟩, ⟨26402⟩⟩
def transferEvent : Nat := 20894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20888 .summary, .result 20710 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20888 .summary)
      LeftBound20722.bound (LeftBound20722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20339⟩⟩) (rawTerms := some (Proof.Events081.exact20888RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20710 .summary)
      LeftBound20705.bound (LeftBound20705.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26401⟩⟩) (rawTerms := some (Proof.Events080.exact20710RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20722.bound, LeftBound20705.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20722.bound, LeftBound20705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20722.actual selector witness, LeftBound20705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20894

namespace LeftBound20898
def owner : Owner := ⟨.program ⟨214⟩, ⟨26403⟩⟩
def transferEvent : Nat := 20898
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20896 .coefficient) (.predecessor 1 20897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20896 .coefficient)
      LeftBound20891.bound (LeftBound20891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20897 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20891.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20891.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20891.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20898

namespace LeftBound20899
def owner : Owner := ⟨.program ⟨214⟩, ⟨26403⟩⟩
def transferEvent : Nat := 20899
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6679⟩⟩]⟩ [⟨.result 5855 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5855 .coefficient)
      LeftAuthority5854.bound (LeftAuthority5854.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6679⟩⟩) (rawTerms := some (Proof.Events022.exact5855RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5854.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5854.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5854.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20899

namespace LeftBound20900
def owner : Owner := ⟨.program ⟨214⟩, ⟨26403⟩⟩
def transferEvent : Nat := 20900
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 20895 .summary) (.transfer 20899) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20895 .summary)
      LeftBound20894.bound (LeftBound20894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26402⟩⟩) (rawTerms := some (Proof.Events081.exact20895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20899)
      LeftBound20899.bound (LeftBound20899.actual selector witness) := by
  exact .transfer (LeftBound20899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20894.bound LeftBound20899.bound
def bound : CoeffClass := .finite ⟨4741253940199267499646124032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20894.bound, LeftBound20899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20894.actual selector witness) * (LeftBound20899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20900

namespace LeftBound20907
def owner : Owner := ⟨.program ⟨214⟩, ⟨74⟩⟩
def transferEvent : Nat := 20907
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 20906 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20906 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20907

namespace LeftBound20911
def owner : Owner := ⟨.program ⟨214⟩, ⟨6630⟩⟩
def transferEvent : Nat := 20911
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 20909 .coefficient) (.predecessor 1 20910 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20909 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20910 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound20911

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
