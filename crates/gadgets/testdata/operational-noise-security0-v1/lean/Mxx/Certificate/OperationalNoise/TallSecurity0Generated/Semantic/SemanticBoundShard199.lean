import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard198

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29710
def owner : Owner := ⟨.program ⟨214⟩, ⟨19039⟩⟩
def transferEvent : Nat := 29710
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19036⟩⟩]⟩ [⟨.result 29702 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29702 .coefficient)
      LeftAuthority29701.bound (LeftAuthority29701.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19036⟩⟩) (rawTerms := some (Proof.Events116.exact29702RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29701.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29701.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29701.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29710

namespace LeftBound29711
def owner : Owner := ⟨.program ⟨214⟩, ⟨19039⟩⟩
def transferEvent : Nat := 29711
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 29710) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29710)
      LeftBound29710.bound (LeftBound29710.actual selector witness) := by
  exact .transfer (LeftBound29710.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound29710.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound29710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound29710.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29711

namespace LeftBound29790
def owner : Owner := ⟨.program ⟨214⟩, ⟨10505⟩⟩
def transferEvent : Nat := 29790
def frameStart : Nat := 29761
def rule : BoundRule := .product (.predecessor 0 29788 .coefficient) (.predecessor 1 29789 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29788 .coefficient)
      LeftAuthority29786.bound (LeftAuthority29786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29786.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29789 .coefficient)
      LeftAuthority29783.bound (LeftAuthority29783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29783.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29783.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29786.bound LeftAuthority29783.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29786.bound, LeftAuthority29783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29786.actual selector witness) * (LeftAuthority29783.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29790

namespace LeftBound29794
def owner : Owner := ⟨.program ⟨214⟩, ⟨10506⟩⟩
def transferEvent : Nat := 29794
def frameStart : Nat := 29761
def rule : BoundRule := .identity (.predecessor 0 29793 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29793 .coefficient)
      LeftBound29790.bound (LeftBound29790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29790.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29790.derived selector witness)

def rawBound : CoeffClass := LeftBound29790.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29790.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29794

namespace LeftBound29811
def owner : Owner := ⟨.program ⟨214⟩, ⟨10588⟩⟩
def transferEvent : Nat := 29811
def frameStart : Nat := 29761
def rule : BoundRule := .sum [.predecessor 0 29809 .coefficient, .predecessor 1 29810 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29809 .coefficient)
      LeftBound29794.bound (LeftBound29794.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29794.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29810 .coefficient)
      LeftAuthority29807.bound (LeftAuthority29807.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29807.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29794.bound, LeftAuthority29807.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29794.bound, LeftAuthority29807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29794.actual selector witness, LeftAuthority29807.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29811

namespace LeftBound29814
def owner : Owner := ⟨.program ⟨214⟩, ⟨10589⟩⟩
def transferEvent : Nat := 29814
def frameStart : Nat := 29761
def rule : BoundRule := .identity (.predecessor 0 29813 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29813 .coefficient)
      LeftBound29811.bound (LeftBound29811.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29811.derived selector witness)

def rawBound : CoeffClass := LeftBound29811.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound29811.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29814

namespace LeftBound29820
def owner : Owner := ⟨.program ⟨214⟩, ⟨10590⟩⟩
def transferEvent : Nat := 29820
def frameStart : Nat := 29761
def rule : BoundRule := .product (.predecessor 0 29818 .coefficient) (.predecessor 1 29819 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29818 .coefficient)
      LeftAuthority29816.bound (LeftAuthority29816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29819 .coefficient)
      LeftBound29814.bound (LeftBound29814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29814.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority29816.bound LeftBound29814.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29816.bound, LeftBound29814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority29816.actual selector witness) * (LeftBound29814.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29820

namespace LeftBound29836
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 29836
def frameStart : Nat := 29761
def rule : BoundRule := .scale (.predecessor 0 29834 .coefficient) (.value (.predecessor 1 29835 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29834 .coefficient)
      LeftAuthority29832.bound (LeftAuthority29832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29832.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29832.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29835 .coefficient)
      LeftAuthority29823.bound (LeftAuthority29823.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29823.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29832.bound LeftAuthority29823.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29832.bound, LeftAuthority29823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29832.actual selector witness) * (LeftAuthority29823.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29836

namespace LeftBound29839
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 29839
def frameStart : Nat := 29761
def rule : BoundRule := .identity (.predecessor 0 29838 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29838 .coefficient)
      LeftAuthority29826.bound (LeftAuthority29826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29826.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29826.derived selector witness)

def rawBound : CoeffClass := LeftAuthority29826.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority29826.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29839

namespace LeftBound29843
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 29843
def frameStart : Nat := 29761
def rule : BoundRule := .product (.predecessor 0 29841 .coefficient) (.predecessor 1 29842 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29841 .coefficient)
      LeftBound29839.bound (LeftBound29839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29842 .coefficient)
      LeftBound29836.bound (LeftBound29836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29836.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29839.bound LeftBound29836.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29839.bound, LeftBound29836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29839.actual selector witness) * (LeftBound29836.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29843

namespace LeftBound29848
def owner : Owner := ⟨.program ⟨214⟩, ⟨10591⟩⟩
def transferEvent : Nat := 29848
def frameStart : Nat := 29761
def rule : BoundRule := .sum [.predecessor 0 29846 .coefficient, .predecessor 1 29847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29846 .coefficient)
      LeftBound29843.bound (LeftBound29843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29847 .coefficient)
      LeftBound29820.bound (LeftBound29820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29820.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29843.bound, LeftBound29820.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29843.bound, LeftBound29820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29843.actual selector witness, LeftBound29820.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29848

namespace LeftBound29852
def owner : Owner := ⟨.program ⟨214⟩, ⟨24929⟩⟩
def transferEvent : Nat := 29852
def frameStart : Nat := 29761
def rule : BoundRule := .product (.predecessor 0 29850 .coefficient) (.predecessor 1 29851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29850 .coefficient)
      LeftBound29848.bound (LeftBound29848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29851 .coefficient)
      LeftAuthority29805.bound (LeftAuthority29805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29805.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29805.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29848.bound LeftAuthority29805.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29848.bound, LeftAuthority29805.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29848.actual selector witness) * (LeftAuthority29805.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29852

namespace LeftBound29863
def owner : Owner := ⟨.program ⟨214⟩, ⟨14806⟩⟩
def transferEvent : Nat := 29863
def frameStart : Nat := 29761
def rule : BoundRule := .product (.predecessor 0 29861 .coefficient) (.predecessor 1 29862 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29861 .coefficient)
      LeftAuthority29816.bound (LeftAuthority29816.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29817RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29816.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29862 .coefficient)
      LeftAuthority29859.bound (LeftAuthority29859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29859.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29816.bound LeftAuthority29859.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29816.bound, LeftAuthority29859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29816.actual selector witness) * (LeftAuthority29859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29863

namespace LeftBound29871
def owner : Owner := ⟨.program ⟨214⟩, ⟨14807⟩⟩
def transferEvent : Nat := 29871
def frameStart : Nat := 29761
def rule : BoundRule := .sum [.predecessor 0 29869 .coefficient, .predecessor 1 29870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29869 .coefficient)
      LeftAuthority29867.bound (LeftAuthority29867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29870 .coefficient)
      LeftBound29863.bound (LeftBound29863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29867.bound, LeftBound29863.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29867.bound, LeftBound29863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29867.actual selector witness, LeftBound29863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29871

namespace LeftBound29875
def owner : Owner := ⟨.program ⟨214⟩, ⟨24930⟩⟩
def transferEvent : Nat := 29875
def frameStart : Nat := 29761
def rule : BoundRule := .sum [.predecessor 0 29873 .coefficient, .predecessor 1 29874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29873 .coefficient)
      LeftBound29871.bound (LeftBound29871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29874 .coefficient)
      LeftBound29852.bound (LeftBound29852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29871.bound, LeftBound29852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29871.bound, LeftBound29852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29871.actual selector witness, LeftBound29852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29875

namespace LeftBound29888
def owner : Owner := ⟨.program ⟨214⟩, ⟨24928⟩⟩
def transferEvent : Nat := 29888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29886 .coefficient, .predecessor 1 29887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29886 .coefficient)
      LeftBound29709.bound (LeftBound29709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29887 .coefficient)
      LeftBound29692.bound (LeftBound29692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29709.bound, LeftBound29692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29709.bound, LeftBound29692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29709.actual selector witness, LeftBound29692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29888

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
