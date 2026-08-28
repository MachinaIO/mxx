import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard093

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound14707
def owner : Owner := ⟨.program ⟨214⟩, ⟨7835⟩⟩
def transferEvent : Nat := 14707
def frameStart : Nat := 14632
def rule : BoundRule := .scale (.predecessor 0 14705 .coefficient) (.value (.predecessor 1 14706 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14705 .coefficient)
      LeftAuthority14703.bound (LeftAuthority14703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14703.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14706 .coefficient)
      LeftAuthority14694.bound (LeftAuthority14694.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority14694.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority14703.bound LeftAuthority14694.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14703.bound, LeftAuthority14694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14703.actual selector witness) * (LeftAuthority14694.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14707

namespace LeftBound14710
def owner : Owner := ⟨.program ⟨214⟩, ⟨6782⟩⟩
def transferEvent : Nat := 14710
def frameStart : Nat := 14632
def rule : BoundRule := .identity (.predecessor 0 14709 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14709 .coefficient)
      LeftAuthority14697.bound (LeftAuthority14697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14697.derived selector witness)

def rawBound : CoeffClass := LeftAuthority14697.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority14697.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound14710

namespace LeftBound14714
def owner : Owner := ⟨.program ⟨214⟩, ⟨7836⟩⟩
def transferEvent : Nat := 14714
def frameStart : Nat := 14632
def rule : BoundRule := .product (.predecessor 0 14712 .coefficient) (.predecessor 1 14713 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14712 .coefficient)
      LeftBound14710.bound (LeftBound14710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14713 .coefficient)
      LeftBound14707.bound (LeftBound14707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14707.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14710.bound LeftBound14707.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14710.bound, LeftBound14707.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14710.actual selector witness) * (LeftBound14707.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14714

namespace LeftBound14719
def owner : Owner := ⟨.program ⟨214⟩, ⟨10791⟩⟩
def transferEvent : Nat := 14719
def frameStart : Nat := 14632
def rule : BoundRule := .sum [.predecessor 0 14717 .coefficient, .predecessor 1 14718 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14717 .coefficient)
      LeftBound14714.bound (LeftBound14714.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14714.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14718 .coefficient)
      LeftBound14691.bound (LeftBound14691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14691.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14714.bound, LeftBound14691.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14714.bound, LeftBound14691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14714.actual selector witness, LeftBound14691.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14719

namespace LeftBound14723
def owner : Owner := ⟨.program ⟨214⟩, ⟨25011⟩⟩
def transferEvent : Nat := 14723
def frameStart : Nat := 14632
def rule : BoundRule := .product (.predecessor 0 14721 .coefficient) (.predecessor 1 14722 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14721 .coefficient)
      LeftBound14719.bound (LeftBound14719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14722 .coefficient)
      LeftAuthority14676.bound (LeftAuthority14676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14719.bound LeftAuthority14676.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14719.bound, LeftAuthority14676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14719.actual selector witness) * (LeftAuthority14676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14723

namespace LeftBound14734
def owner : Owner := ⟨.program ⟨214⟩, ⟨14971⟩⟩
def transferEvent : Nat := 14734
def frameStart : Nat := 14632
def rule : BoundRule := .product (.predecessor 0 14732 .coefficient) (.predecessor 1 14733 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14732 .coefficient)
      LeftAuthority14687.bound (LeftAuthority14687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14733 .coefficient)
      LeftAuthority14730.bound (LeftAuthority14730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14730.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14730.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority14687.bound LeftAuthority14730.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14687.bound, LeftAuthority14730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority14687.actual selector witness) * (LeftAuthority14730.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14734

namespace LeftBound14742
def owner : Owner := ⟨.program ⟨214⟩, ⟨14972⟩⟩
def transferEvent : Nat := 14742
def frameStart : Nat := 14632
def rule : BoundRule := .sum [.predecessor 0 14740 .coefficient, .predecessor 1 14741 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14740 .coefficient)
      LeftAuthority14738.bound (LeftAuthority14738.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14739RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14738.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14741 .coefficient)
      LeftBound14734.bound (LeftBound14734.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14734.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14734.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority14738.bound, LeftBound14734.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14738.bound, LeftBound14734.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority14738.actual selector witness, LeftBound14734.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14742

namespace LeftBound14746
def owner : Owner := ⟨.program ⟨214⟩, ⟨25012⟩⟩
def transferEvent : Nat := 14746
def frameStart : Nat := 14632
def rule : BoundRule := .sum [.predecessor 0 14744 .coefficient, .predecessor 1 14745 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14744 .coefficient)
      LeftBound14742.bound (LeftBound14742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14745 .coefficient)
      LeftBound14723.bound (LeftBound14723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14723.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14742.bound, LeftBound14723.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14742.bound, LeftBound14723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14742.actual selector witness, LeftBound14723.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14746

namespace LeftBound14759
def owner : Owner := ⟨.program ⟨214⟩, ⟨25010⟩⟩
def transferEvent : Nat := 14759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 14757 .coefficient, .predecessor 1 14758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14757 .coefficient)
      LeftBound14580.bound (LeftBound14580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14580.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14758 .coefficient)
      LeftBound14563.bound (LeftBound14563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14580.bound, LeftBound14563.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14580.bound, LeftBound14563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14580.actual selector witness, LeftBound14563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14759

namespace LeftBound14762
def owner : Owner := ⟨.program ⟨214⟩, ⟨25010⟩⟩
def transferEvent : Nat := 14762
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 14756 .summary, .result 14570 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14756 .summary)
      LeftBound14582.bound (LeftBound14582.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19115⟩⟩) (rawTerms := some (Proof.Events057.exact14756RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14570 .summary)
      LeftBound14565.bound (LeftBound14565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25009⟩⟩) (rawTerms := some (Proof.Events056.exact14570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound14582.bound, LeftBound14565.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14582.bound, LeftBound14565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound14582.actual selector witness, LeftBound14565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound14762

namespace LeftBound14766
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def transferEvent : Nat := 14766
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14764 .coefficient) (.predecessor 1 14765 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14764 .coefficient)
      LeftBound14759.bound (LeftBound14759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14765 .coefficient)
      LeftAuthority14466.bound (LeftAuthority14466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14759.bound LeftAuthority14466.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14759.bound, LeftAuthority14466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14759.actual selector witness) * (LeftAuthority14466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14766

namespace LeftBound14767
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def transferEvent : Nat := 14767
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26616⟩⟩]⟩ [⟨.result 14467 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14467 .coefficient)
      LeftAuthority14466.bound (LeftAuthority14466.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26616⟩⟩) (rawTerms := some (Proof.Events056.exact14467RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14466.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14466.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14466.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14466.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14767

namespace LeftBound14768
def owner : Owner := ⟨.program ⟨214⟩, ⟨26618⟩⟩
def transferEvent : Nat := 14768
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 14763 .summary) (.transfer 14767) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14763 .summary)
      LeftBound14762.bound (LeftBound14762.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25010⟩⟩) (rawTerms := some (Proof.Events057.exact14763RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound14762.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 14767)
      LeftBound14767.bound (LeftBound14767.actual selector witness) := by
  exact .transfer (LeftBound14767.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14762.bound LeftBound14767.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14762.bound, LeftBound14767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14762.actual selector witness) * (LeftBound14767.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14768

namespace LeftBound14779
def owner : Owner := ⟨.program ⟨214⟩, ⟨20554⟩⟩
def transferEvent : Nat := 14779
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 14777 .coefficient) (.value (.predecessor 1 14778 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14777 .coefficient)
      LeftAuthority14775.bound (LeftAuthority14775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14778 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority14775.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14775.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14775.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound14779

namespace LeftBound14783
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def transferEvent : Nat := 14783
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 14781 .coefficient) (.predecessor 1 14782 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 14781 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 14782 .coefficient)
      LeftBound14779.bound (LeftBound14779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14779.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound14779.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound14779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound14779.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound14783

namespace LeftBound14784
def owner : Owner := ⟨.program ⟨214⟩, ⟨20555⟩⟩
def transferEvent : Nat := 14784
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20552⟩⟩]⟩ [⟨.result 14776 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14776 .coefficient)
      LeftAuthority14775.bound (LeftAuthority14775.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20552⟩⟩) (rawTerms := some (Proof.Events057.exact14776RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14775.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14775.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14775.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound14784

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
