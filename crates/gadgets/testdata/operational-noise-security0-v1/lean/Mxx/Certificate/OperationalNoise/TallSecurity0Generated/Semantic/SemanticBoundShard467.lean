import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard466

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68764
def owner : Owner := ⟨.program ⟨214⟩, ⟨19743⟩⟩
def transferEvent : Nat := 68764
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68762 .coefficient) (.predecessor 1 68763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68762 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68763 .coefficient)
      LeftBound68760.bound (LeftBound68760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68760.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound68760.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound68760.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound68760.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68764

namespace LeftBound68765
def owner : Owner := ⟨.program ⟨214⟩, ⟨19743⟩⟩
def transferEvent : Nat := 68765
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19740⟩⟩]⟩ [⟨.result 68757 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68757 .coefficient)
      LeftAuthority68756.bound (LeftAuthority68756.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19740⟩⟩) (rawTerms := some (Proof.Events268.exact68757RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68756.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68756.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68756.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68765

namespace LeftBound68766
def owner : Owner := ⟨.program ⟨214⟩, ⟨19743⟩⟩
def transferEvent : Nat := 68766
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 68765) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68765)
      LeftBound68765.bound (LeftBound68765.actual selector witness) := by
  exact .transfer (LeftBound68765.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound68765.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound68765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound68765.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68766

namespace LeftBound68845
def owner : Owner := ⟨.program ⟨214⟩, ⟨11754⟩⟩
def transferEvent : Nat := 68845
def frameStart : Nat := 68816
def rule : BoundRule := .product (.predecessor 0 68843 .coefficient) (.predecessor 1 68844 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68843 .coefficient)
      LeftAuthority68841.bound (LeftAuthority68841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68841.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68844 .coefficient)
      LeftAuthority68838.bound (LeftAuthority68838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68841.bound LeftAuthority68838.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68841.bound, LeftAuthority68838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68841.actual selector witness) * (LeftAuthority68838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68845

namespace LeftBound68849
def owner : Owner := ⟨.program ⟨214⟩, ⟨11755⟩⟩
def transferEvent : Nat := 68849
def frameStart : Nat := 68816
def rule : BoundRule := .identity (.predecessor 0 68848 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68848 .coefficient)
      LeftBound68845.bound (LeftBound68845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68845.derived selector witness)

def rawBound : CoeffClass := LeftBound68845.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68845.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68849

namespace LeftBound68866
def owner : Owner := ⟨.program ⟨214⟩, ⟨11853⟩⟩
def transferEvent : Nat := 68866
def frameStart : Nat := 68816
def rule : BoundRule := .sum [.predecessor 0 68864 .coefficient, .predecessor 1 68865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68864 .coefficient)
      LeftBound68849.bound (LeftBound68849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68865 .coefficient)
      LeftAuthority68862.bound (LeftAuthority68862.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68849.bound, LeftAuthority68862.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68849.bound, LeftAuthority68862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68849.actual selector witness, LeftAuthority68862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68866

namespace LeftBound68869
def owner : Owner := ⟨.program ⟨214⟩, ⟨11854⟩⟩
def transferEvent : Nat := 68869
def frameStart : Nat := 68816
def rule : BoundRule := .identity (.predecessor 0 68868 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68868 .coefficient)
      LeftBound68866.bound (LeftBound68866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68866.derived selector witness)

def rawBound : CoeffClass := LeftBound68866.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68866.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68869

namespace LeftBound68875
def owner : Owner := ⟨.program ⟨214⟩, ⟨11855⟩⟩
def transferEvent : Nat := 68875
def frameStart : Nat := 68816
def rule : BoundRule := .product (.predecessor 0 68873 .coefficient) (.predecessor 1 68874 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68873 .coefficient)
      LeftAuthority68871.bound (LeftAuthority68871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68874 .coefficient)
      LeftBound68869.bound (LeftBound68869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority68871.bound LeftBound68869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68871.bound, LeftBound68869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority68871.actual selector witness) * (LeftBound68869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68875

namespace LeftBound68891
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 68891
def frameStart : Nat := 68816
def rule : BoundRule := .scale (.predecessor 0 68889 .coefficient) (.value (.predecessor 1 68890 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68889 .coefficient)
      LeftAuthority68887.bound (LeftAuthority68887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68890 .coefficient)
      LeftAuthority68878.bound (LeftAuthority68878.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68878.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68887.bound LeftAuthority68878.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68887.bound, LeftAuthority68878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68887.actual selector witness) * (LeftAuthority68878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68891

namespace LeftBound68894
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 68894
def frameStart : Nat := 68816
def rule : BoundRule := .identity (.predecessor 0 68893 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68893 .coefficient)
      LeftAuthority68881.bound (LeftAuthority68881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68881.derived selector witness)

def rawBound : CoeffClass := LeftAuthority68881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority68881.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68894

namespace LeftBound68898
def owner : Owner := ⟨.program ⟨214⟩, ⟨7863⟩⟩
def transferEvent : Nat := 68898
def frameStart : Nat := 68816
def rule : BoundRule := .product (.predecessor 0 68896 .coefficient) (.predecessor 1 68897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68896 .coefficient)
      LeftBound68894.bound (LeftBound68894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68897 .coefficient)
      LeftBound68891.bound (LeftBound68891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68894.bound LeftBound68891.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68894.bound, LeftBound68891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68894.actual selector witness) * (LeftBound68891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68898

namespace LeftBound68903
def owner : Owner := ⟨.program ⟨214⟩, ⟨11856⟩⟩
def transferEvent : Nat := 68903
def frameStart : Nat := 68816
def rule : BoundRule := .sum [.predecessor 0 68901 .coefficient, .predecessor 1 68902 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68901 .coefficient)
      LeftBound68898.bound (LeftBound68898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68902 .coefficient)
      LeftBound68875.bound (LeftBound68875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68898.bound, LeftBound68875.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68898.bound, LeftBound68875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68898.actual selector witness, LeftBound68875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68903

namespace LeftBound68907
def owner : Owner := ⟨.program ⟨214⟩, ⟨25140⟩⟩
def transferEvent : Nat := 68907
def frameStart : Nat := 68816
def rule : BoundRule := .product (.predecessor 0 68905 .coefficient) (.predecessor 1 68906 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68905 .coefficient)
      LeftBound68903.bound (LeftBound68903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68906 .coefficient)
      LeftAuthority68860.bound (LeftAuthority68860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events268.exact68861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68903.bound LeftAuthority68860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68903.bound, LeftAuthority68860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68903.actual selector witness) * (LeftAuthority68860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68907

namespace LeftBound68918
def owner : Owner := ⟨.program ⟨214⟩, ⟨16260⟩⟩
def transferEvent : Nat := 68918
def frameStart : Nat := 68816
def rule : BoundRule := .product (.predecessor 0 68916 .coefficient) (.predecessor 1 68917 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68916 .coefficient)
      LeftAuthority68871.bound (LeftAuthority68871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68917 .coefficient)
      LeftAuthority68914.bound (LeftAuthority68914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68871.bound LeftAuthority68914.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68871.bound, LeftAuthority68914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68871.actual selector witness) * (LeftAuthority68914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68918

namespace LeftBound68926
def owner : Owner := ⟨.program ⟨214⟩, ⟨16261⟩⟩
def transferEvent : Nat := 68926
def frameStart : Nat := 68816
def rule : BoundRule := .sum [.predecessor 0 68924 .coefficient, .predecessor 1 68925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68924 .coefficient)
      LeftAuthority68922.bound (LeftAuthority68922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68925 .coefficient)
      LeftBound68918.bound (LeftBound68918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68922.bound, LeftBound68918.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68922.bound, LeftBound68918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68922.actual selector witness, LeftBound68918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68926

namespace LeftBound68930
def owner : Owner := ⟨.program ⟨214⟩, ⟨25141⟩⟩
def transferEvent : Nat := 68930
def frameStart : Nat := 68816
def rule : BoundRule := .sum [.predecessor 0 68928 .coefficient, .predecessor 1 68929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68928 .coefficient)
      LeftBound68926.bound (LeftBound68926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68929 .coefficient)
      LeftBound68907.bound (LeftBound68907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68926.bound, LeftBound68907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68926.bound, LeftBound68907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68926.actual selector witness, LeftBound68907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68930

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
