import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard693

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100813
def owner : Owner := ⟨.program ⟨214⟩, ⟨15455⟩⟩
def transferEvent : Nat := 100813
def frameStart : Nat := 100766
def rule : BoundRule := .identity (.predecessor 0 100812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100812 .coefficient)
      LeftBound100810.bound (LeftBound100810.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100810.derived selector witness)

def rawBound : CoeffClass := LeftBound100810.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound100810.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100813

namespace LeftBound100819
def owner : Owner := ⟨.program ⟨214⟩, ⟨15456⟩⟩
def transferEvent : Nat := 100819
def frameStart : Nat := 100766
def rule : BoundRule := .product (.predecessor 0 100817 .coefficient) (.predecessor 1 100818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100817 .coefficient)
      LeftAuthority100815.bound (LeftAuthority100815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100818 .coefficient)
      LeftBound100813.bound (LeftBound100813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority100815.bound LeftBound100813.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100815.bound, LeftBound100813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority100815.actual selector witness) * (LeftBound100813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100819

namespace LeftBound100827
def owner : Owner := ⟨.program ⟨214⟩, ⟨15457⟩⟩
def transferEvent : Nat := 100827
def frameStart : Nat := 100766
def rule : BoundRule := .sum [.predecessor 0 100825 .coefficient, .predecessor 1 100826 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100825 .coefficient)
      LeftAuthority100823.bound (LeftAuthority100823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100823.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100826 .coefficient)
      LeftBound100819.bound (LeftBound100819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100819.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100823.bound, LeftBound100819.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100823.bound, LeftBound100819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100823.actual selector witness, LeftBound100819.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100827

namespace LeftBound100831
def owner : Owner := ⟨.program ⟨214⟩, ⟨26964⟩⟩
def transferEvent : Nat := 100831
def frameStart : Nat := 100766
def rule : BoundRule := .product (.predecessor 0 100829 .coefficient) (.predecessor 1 100830 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100829 .coefficient)
      LeftBound100827.bound (LeftBound100827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100830 .coefficient)
      LeftAuthority100804.bound (LeftAuthority100804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100804.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100827.bound LeftAuthority100804.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100827.bound, LeftAuthority100804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100827.actual selector witness) * (LeftAuthority100804.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100831

namespace LeftBound100842
def owner : Owner := ⟨.program ⟨214⟩, ⟨17309⟩⟩
def transferEvent : Nat := 100842
def frameStart : Nat := 100766
def rule : BoundRule := .product (.predecessor 0 100840 .coefficient) (.predecessor 1 100841 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100840 .coefficient)
      LeftAuthority100815.bound (LeftAuthority100815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100841 .coefficient)
      LeftAuthority100838.bound (LeftAuthority100838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100815.bound LeftAuthority100838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100815.bound, LeftAuthority100838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority100815.actual selector witness) * (LeftAuthority100838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100842

namespace LeftBound100850
def owner : Owner := ⟨.program ⟨214⟩, ⟨17310⟩⟩
def transferEvent : Nat := 100850
def frameStart : Nat := 100766
def rule : BoundRule := .sum [.predecessor 0 100848 .coefficient, .predecessor 1 100849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100848 .coefficient)
      LeftAuthority100846.bound (LeftAuthority100846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100849 .coefficient)
      LeftBound100842.bound (LeftBound100842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100846.bound, LeftBound100842.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100846.bound, LeftBound100842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100846.actual selector witness, LeftBound100842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100850

namespace LeftBound100854
def owner : Owner := ⟨.program ⟨214⟩, ⟨26968⟩⟩
def transferEvent : Nat := 100854
def frameStart : Nat := 100766
def rule : BoundRule := .sum [.predecessor 0 100852 .coefficient, .predecessor 1 100853 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100852 .coefficient)
      LeftBound100850.bound (LeftBound100850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100853 .coefficient)
      LeftBound100831.bound (LeftBound100831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100831.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100850.bound, LeftBound100831.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100850.bound, LeftBound100831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100850.actual selector witness, LeftBound100831.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100854

namespace LeftBound100867
def owner : Owner := ⟨.program ⟨214⟩, ⟨26966⟩⟩
def transferEvent : Nat := 100867
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100865 .coefficient, .predecessor 1 100866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100865 .coefficient)
      LeftBound100720.bound (LeftBound100720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100866 .coefficient)
      LeftBound100703.bound (LeftBound100703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100720.bound, LeftBound100703.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100720.bound, LeftBound100703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100720.actual selector witness, LeftBound100703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100867

namespace LeftBound100870
def owner : Owner := ⟨.program ⟨214⟩, ⟨26966⟩⟩
def transferEvent : Nat := 100870
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100864 .summary, .result 100710 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100864 .summary)
      LeftBound100722.bound (LeftBound100722.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20816⟩⟩) (rawTerms := some (Proof.Events394.exact100864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100710 .summary)
      LeftBound100705.bound (LeftBound100705.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26965⟩⟩) (rawTerms := some (Proof.Events393.exact100710RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100705.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100722.bound, LeftBound100705.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100722.bound, LeftBound100705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100722.actual selector witness, LeftBound100705.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100870

namespace LeftBound100894
def owner : Owner := ⟨.program ⟨214⟩, ⟨10956⟩⟩
def transferEvent : Nat := 100894
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 100892 .coefficient) (.predecessor 1 100893 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100892 .coefficient)
      LeftAuthority4910.bound (LeftAuthority4910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100893 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4910.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4910.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4910.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100894

namespace LeftBound100899
def owner : Owner := ⟨.program ⟨214⟩, ⟨7111⟩⟩
def transferEvent : Nat := 100899
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100897 .coefficient) (.predecessor 1 100898 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100897 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100898 .coefficient)
      LeftBound13986.bound (LeftBound13986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound13986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound13986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound13986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100899

namespace LeftBound100904
def owner : Owner := ⟨.program ⟨214⟩, ⟨10957⟩⟩
def transferEvent : Nat := 100904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100902 .coefficient, .predecessor 1 100903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100902 .coefficient)
      LeftBound100899.bound (LeftBound100899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100903 .coefficient)
      LeftBound100894.bound (LeftBound100894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100894.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100899.bound, LeftBound100894.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100899.bound, LeftBound100894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100899.actual selector witness, LeftBound100894.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100904

namespace LeftBound100908
def owner : Owner := ⟨.program ⟨214⟩, ⟨10958⟩⟩
def transferEvent : Nat := 100908
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100906 .coefficient, .predecessor 1 100907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100906 .coefficient)
      LeftBound100904.bound (LeftBound100904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100907 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100904.bound, LeftBound13978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100904.bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100904.actual selector witness, LeftBound13978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100908

namespace LeftBound100909
def owner : Owner := ⟨.program ⟨214⟩, ⟨10958⟩⟩
def transferEvent : Nat := 100909
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩ [⟨.result 13979 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13979 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨88⟩⟩) (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13978.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13978.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100909

namespace LeftBound100914
def owner : Owner := ⟨.program ⟨214⟩, ⟨10959⟩⟩
def transferEvent : Nat := 100914
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100912 .coefficient) (.predecessor 1 100913 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100912 .coefficient)
      LeftBound100908.bound (LeftBound100908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100913 .coefficient)
      LeftAuthority4913.bound (LeftAuthority4913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound100908.bound LeftAuthority4913.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100908.bound, LeftAuthority4913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound100908.actual selector witness) * (LeftAuthority4913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100914

namespace LeftBound100915
def owner : Owner := ⟨.program ⟨214⟩, ⟨10959⟩⟩
def transferEvent : Nat := 100915
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10827⟩⟩], []⟩ [⟨.result 4914 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4914 .coefficient)
      LeftAuthority4913.bound (LeftAuthority4913.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10827⟩⟩) (rawTerms := some (Proof.Events019.exact4914RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4913.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4913.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4913.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100915

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
