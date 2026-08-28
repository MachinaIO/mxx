import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard684
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard685

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99815
def owner : Owner := ⟨.program ⟨214⟩, ⟨25903⟩⟩
def transferEvent : Nat := 99815
def frameStart : Nat := 99713
def rule : BoundRule := .sum [.predecessor 0 99813 .coefficient, .predecessor 1 99814 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99813 .coefficient)
      LeftBound99811.bound (LeftBound99811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99811.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99814 .coefficient)
      LeftBound99792.bound (LeftBound99792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99797RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99792.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99811.bound, LeftBound99792.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99811.bound, LeftBound99792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99811.actual selector witness, LeftBound99792.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99815

namespace LeftBound99828
def owner : Owner := ⟨.program ⟨214⟩, ⟨25901⟩⟩
def transferEvent : Nat := 99828
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99826 .coefficient, .predecessor 1 99827 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99826 .coefficient)
      LeftBound99673.bound (LeftBound99673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99673.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99827 .coefficient)
      LeftBound99656.bound (LeftBound99656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99673.bound, LeftBound99656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99673.bound, LeftBound99656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99673.actual selector witness, LeftBound99656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99828

namespace LeftBound99831
def owner : Owner := ⟨.program ⟨214⟩, ⟨25901⟩⟩
def transferEvent : Nat := 99831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99825 .summary, .result 99663 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99825 .summary)
      LeftBound99675.bound (LeftBound99675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19376⟩⟩) (rawTerms := some (Proof.Events389.exact99825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99663 .summary)
      LeftBound99658.bound (LeftBound99658.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25900⟩⟩) (rawTerms := some (Proof.Events389.exact99663RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99675.bound, LeftBound99658.bound]
def bound : CoeffClass := .finite ⟨352042398396416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99675.bound, LeftBound99658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99675.actual selector witness, LeftBound99658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99831

namespace LeftBound99835
def owner : Owner := ⟨.program ⟨214⟩, ⟨27399⟩⟩
def transferEvent : Nat := 99835
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99833 .coefficient) (.predecessor 1 99834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99833 .coefficient)
      LeftBound99828.bound (LeftBound99828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99834 .coefficient)
      LeftAuthority99578.bound (LeftAuthority99578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events388.exact99579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99578.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99828.bound LeftAuthority99578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99828.bound, LeftAuthority99578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99828.actual selector witness) * (LeftAuthority99578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99835

namespace LeftBound99836
def owner : Owner := ⟨.program ⟨214⟩, ⟨27399⟩⟩
def transferEvent : Nat := 99836
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27397⟩⟩]⟩ [⟨.result 99579 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99579 .coefficient)
      LeftAuthority99578.bound (LeftAuthority99578.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27397⟩⟩) (rawTerms := some (Proof.Events388.exact99579RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99578.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99578.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99578.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99578.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99836

namespace LeftBound99837
def owner : Owner := ⟨.program ⟨214⟩, ⟨27399⟩⟩
def transferEvent : Nat := 99837
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99832 .summary) (.transfer 99836) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99832 .summary)
      LeftBound99831.bound (LeftBound99831.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25901⟩⟩) (rawTerms := some (Proof.Events389.exact99832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99836)
      LeftBound99836.bound (LeftBound99836.actual selector witness) := by
  exact .transfer (LeftBound99836.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99831.bound LeftBound99836.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99831.bound, LeftBound99836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99831.actual selector witness) * (LeftBound99836.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99837

namespace LeftBound99848
def owner : Owner := ⟨.program ⟨214⟩, ⟨21103⟩⟩
def transferEvent : Nat := 99848
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 99846 .coefficient) (.value (.predecessor 1 99847 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99846 .coefficient)
      LeftAuthority99844.bound (LeftAuthority99844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99847 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority99844.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99844.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99844.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99848

namespace LeftBound99852
def owner : Owner := ⟨.program ⟨214⟩, ⟨21104⟩⟩
def transferEvent : Nat := 99852
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99850 .coefficient) (.predecessor 1 99851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99850 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99851 .coefficient)
      LeftBound99848.bound (LeftBound99848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99848.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound99848.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound99848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound99848.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99852

namespace LeftBound99853
def owner : Owner := ⟨.program ⟨214⟩, ⟨21104⟩⟩
def transferEvent : Nat := 99853
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21101⟩⟩]⟩ [⟨.result 99845 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99845 .coefficient)
      LeftAuthority99844.bound (LeftAuthority99844.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21101⟩⟩) (rawTerms := some (Proof.Events390.exact99845RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99844.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99844.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99844.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99853

namespace LeftBound99854
def owner : Owner := ⟨.program ⟨214⟩, ⟨21104⟩⟩
def transferEvent : Nat := 99854
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 99853) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99853)
      LeftBound99853.bound (LeftBound99853.actual selector witness) := by
  exact .transfer (LeftBound99853.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound99853.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound99853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound99853.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99854

namespace LeftBound99925
def owner : Owner := ⟨.program ⟨214⟩, ⟨15693⟩⟩
def transferEvent : Nat := 99925
def frameStart : Nat := 99898
def rule : BoundRule := .identity (.predecessor 0 99924 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99924 .coefficient)
      LeftAuthority99922.bound (LeftAuthority99922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99922.derived selector witness)

def rawBound : CoeffClass := LeftAuthority99922.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority99922.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99925

namespace LeftBound99942
def owner : Owner := ⟨.program ⟨214⟩, ⟨15769⟩⟩
def transferEvent : Nat := 99942
def frameStart : Nat := 99898
def rule : BoundRule := .sum [.predecessor 0 99940 .coefficient, .predecessor 1 99941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99940 .coefficient)
      LeftBound99925.bound (LeftBound99925.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99941 .coefficient)
      LeftAuthority99938.bound (LeftAuthority99938.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority99938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99925.bound, LeftAuthority99938.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99925.bound, LeftAuthority99938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99925.actual selector witness, LeftAuthority99938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99942

namespace LeftBound99945
def owner : Owner := ⟨.program ⟨214⟩, ⟨15770⟩⟩
def transferEvent : Nat := 99945
def frameStart : Nat := 99898
def rule : BoundRule := .identity (.predecessor 0 99944 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99944 .coefficient)
      LeftBound99942.bound (LeftBound99942.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound99942.derived selector witness)

def rawBound : CoeffClass := LeftBound99942.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound99942.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound99945

namespace LeftBound99951
def owner : Owner := ⟨.program ⟨214⟩, ⟨15771⟩⟩
def transferEvent : Nat := 99951
def frameStart : Nat := 99898
def rule : BoundRule := .product (.predecessor 0 99949 .coefficient) (.predecessor 1 99950 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99949 .coefficient)
      LeftAuthority99947.bound (LeftAuthority99947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99950 .coefficient)
      LeftBound99945.bound (LeftBound99945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99945.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority99947.bound LeftBound99945.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99947.bound, LeftBound99945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority99947.actual selector witness) * (LeftBound99945.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99951

namespace LeftBound99959
def owner : Owner := ⟨.program ⟨214⟩, ⟨15772⟩⟩
def transferEvent : Nat := 99959
def frameStart : Nat := 99898
def rule : BoundRule := .sum [.predecessor 0 99957 .coefficient, .predecessor 1 99958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99957 .coefficient)
      LeftAuthority99955.bound (LeftAuthority99955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99955.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99955.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99958 .coefficient)
      LeftBound99951.bound (LeftBound99951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99955.bound, LeftBound99951.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99955.bound, LeftBound99951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99955.actual selector witness, LeftBound99951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99959

namespace LeftBound99963
def owner : Owner := ⟨.program ⟨214⟩, ⟨27398⟩⟩
def transferEvent : Nat := 99963
def frameStart : Nat := 99898
def rule : BoundRule := .product (.predecessor 0 99961 .coefficient) (.predecessor 1 99962 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99961 .coefficient)
      LeftBound99959.bound (LeftBound99959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99962 .coefficient)
      LeftAuthority99936.bound (LeftAuthority99936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99936.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99959.bound LeftAuthority99936.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99959.bound, LeftAuthority99936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99959.actual selector witness) * (LeftAuthority99936.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99963

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
