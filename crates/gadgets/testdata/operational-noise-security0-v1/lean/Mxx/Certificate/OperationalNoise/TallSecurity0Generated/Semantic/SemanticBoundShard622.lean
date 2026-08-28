import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard621

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91693
def owner : Owner := ⟨.program ⟨214⟩, ⟨21763⟩⟩
def transferEvent : Nat := 91693
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21760⟩⟩]⟩ [⟨.result 91685 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91685 .coefficient)
      LeftAuthority91684.bound (LeftAuthority91684.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21760⟩⟩) (rawTerms := some (Proof.Events358.exact91685RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91684.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91684.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91684.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91693

namespace LeftBound91694
def owner : Owner := ⟨.program ⟨214⟩, ⟨21763⟩⟩
def transferEvent : Nat := 91694
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 91693) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91693)
      LeftBound91693.bound (LeftBound91693.actual selector witness) := by
  exact .transfer (LeftBound91693.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound91693.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound91693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound91693.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91694

namespace LeftBound91789
def owner : Owner := ⟨.program ⟨214⟩, ⟨16263⟩⟩
def transferEvent : Nat := 91789
def frameStart : Nat := 91750
def rule : BoundRule := .identity (.predecessor 0 91788 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91788 .coefficient)
      LeftAuthority91786.bound (LeftAuthority91786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91786.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91786.derived selector witness)

def rawBound : CoeffClass := LeftAuthority91786.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority91786.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91789

namespace LeftBound91806
def owner : Owner := ⟨.program ⟨214⟩, ⟨16337⟩⟩
def transferEvent : Nat := 91806
def frameStart : Nat := 91750
def rule : BoundRule := .sum [.predecessor 0 91804 .coefficient, .predecessor 1 91805 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91804 .coefficient)
      LeftBound91789.bound (LeftBound91789.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91805 .coefficient)
      LeftAuthority91802.bound (LeftAuthority91802.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority91802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91789.bound, LeftAuthority91802.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91789.bound, LeftAuthority91802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91789.actual selector witness, LeftAuthority91802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91806

namespace LeftBound91809
def owner : Owner := ⟨.program ⟨214⟩, ⟨16338⟩⟩
def transferEvent : Nat := 91809
def frameStart : Nat := 91750
def rule : BoundRule := .identity (.predecessor 0 91808 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91808 .coefficient)
      LeftBound91806.bound (LeftBound91806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91806.derived selector witness)

def rawBound : CoeffClass := LeftBound91806.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound91806.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91809

namespace LeftBound91815
def owner : Owner := ⟨.program ⟨214⟩, ⟨16339⟩⟩
def transferEvent : Nat := 91815
def frameStart : Nat := 91750
def rule : BoundRule := .product (.predecessor 0 91813 .coefficient) (.predecessor 1 91814 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91813 .coefficient)
      LeftAuthority91811.bound (LeftAuthority91811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91814 .coefficient)
      LeftBound91809.bound (LeftBound91809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91809.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority91811.bound LeftBound91809.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91811.bound, LeftBound91809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority91811.actual selector witness) * (LeftBound91809.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91815

namespace LeftBound91823
def owner : Owner := ⟨.program ⟨214⟩, ⟨16340⟩⟩
def transferEvent : Nat := 91823
def frameStart : Nat := 91750
def rule : BoundRule := .sum [.predecessor 0 91821 .coefficient, .predecessor 1 91822 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91821 .coefficient)
      LeftAuthority91819.bound (LeftAuthority91819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91822 .coefficient)
      LeftBound91815.bound (LeftBound91815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91819.bound, LeftBound91815.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91819.bound, LeftBound91815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91819.actual selector witness, LeftBound91815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91823

namespace LeftBound91827
def owner : Owner := ⟨.program ⟨214⟩, ⟨28511⟩⟩
def transferEvent : Nat := 91827
def frameStart : Nat := 91750
def rule : BoundRule := .product (.predecessor 0 91825 .coefficient) (.predecessor 1 91826 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91825 .coefficient)
      LeftBound91823.bound (LeftBound91823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91824RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91826 .coefficient)
      LeftAuthority91800.bound (LeftAuthority91800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91800.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91823.bound LeftAuthority91800.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91823.bound, LeftAuthority91800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91823.actual selector witness) * (LeftAuthority91800.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91827

namespace LeftBound91838
def owner : Owner := ⟨.program ⟨214⟩, ⟨17608⟩⟩
def transferEvent : Nat := 91838
def frameStart : Nat := 91750
def rule : BoundRule := .product (.predecessor 0 91836 .coefficient) (.predecessor 1 91837 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91836 .coefficient)
      LeftAuthority91811.bound (LeftAuthority91811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91811.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91837 .coefficient)
      LeftAuthority91834.bound (LeftAuthority91834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority91811.bound LeftAuthority91834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91811.bound, LeftAuthority91834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority91811.actual selector witness) * (LeftAuthority91834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91838

namespace LeftBound91846
def owner : Owner := ⟨.program ⟨214⟩, ⟨17609⟩⟩
def transferEvent : Nat := 91846
def frameStart : Nat := 91750
def rule : BoundRule := .sum [.predecessor 0 91844 .coefficient, .predecessor 1 91845 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91844 .coefficient)
      LeftAuthority91842.bound (LeftAuthority91842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91845 .coefficient)
      LeftBound91838.bound (LeftBound91838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91842.bound, LeftBound91838.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91842.bound, LeftBound91838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91842.actual selector witness, LeftBound91838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91846

namespace LeftBound91850
def owner : Owner := ⟨.program ⟨214⟩, ⟨28516⟩⟩
def transferEvent : Nat := 91850
def frameStart : Nat := 91750
def rule : BoundRule := .sum [.predecessor 0 91848 .coefficient, .predecessor 1 91849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91848 .coefficient)
      LeftBound91846.bound (LeftBound91846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91849 .coefficient)
      LeftBound91827.bound (LeftBound91827.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91827.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91846.bound, LeftBound91827.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91846.bound, LeftBound91827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91846.actual selector witness, LeftBound91827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91850

namespace LeftBound91863
def owner : Owner := ⟨.program ⟨214⟩, ⟨28513⟩⟩
def transferEvent : Nat := 91863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 91861 .coefficient, .predecessor 1 91862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91861 .coefficient)
      LeftBound91692.bound (LeftBound91692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91862 .coefficient)
      LeftBound91675.bound (LeftBound91675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91692.bound, LeftBound91675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91692.bound, LeftBound91675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91692.actual selector witness, LeftBound91675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91863

namespace LeftBound91866
def owner : Owner := ⟨.program ⟨214⟩, ⟨28513⟩⟩
def transferEvent : Nat := 91866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 91860 .summary, .result 91682 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91860 .summary)
      LeftBound91694.bound (LeftBound91694.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21763⟩⟩) (rawTerms := some (Proof.Events358.exact91860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91682 .summary)
      LeftBound91677.bound (LeftBound91677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28512⟩⟩) (rawTerms := some (Proof.Events358.exact91682RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91694.bound, LeftBound91677.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91694.bound, LeftBound91677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91694.actual selector witness, LeftBound91677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91866

namespace LeftBound91870
def owner : Owner := ⟨.program ⟨214⟩, ⟨28514⟩⟩
def transferEvent : Nat := 91870
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91868 .coefficient) (.predecessor 1 91869 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91868 .coefficient)
      LeftBound91863.bound (LeftBound91863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events358.exact91867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91869 .coefficient)
      LeftBound5658.bound (LeftBound5658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91863.bound LeftBound5658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91863.bound, LeftBound5658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91863.actual selector witness) * (LeftBound5658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91870

namespace LeftBound91871
def owner : Owner := ⟨.program ⟨214⟩, ⟨28514⟩⟩
def transferEvent : Nat := 91871
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩ [⟨.result 5655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5655 .coefficient)
      LeftAuthority5654.bound (LeftAuthority5654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6677⟩⟩) (rawTerms := some (Proof.Events022.exact5655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91871

namespace LeftBound91872
def owner : Owner := ⟨.program ⟨214⟩, ⟨28514⟩⟩
def transferEvent : Nat := 91872
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91867 .summary) (.transfer 91871) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91867 .summary)
      LeftBound91866.bound (LeftBound91866.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28513⟩⟩) (rawTerms := some (Proof.Events358.exact91867RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91866.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91871)
      LeftBound91871.bound (LeftBound91871.actual selector witness) := by
  exact .transfer (LeftBound91871.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91866.bound LeftBound91871.bound
def bound : CoeffClass := .finite ⟨4742405496644812892115304448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91866.bound, LeftBound91871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91866.actual selector witness) * (LeftBound91871.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91872

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
