import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard700

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106797
def owner : Owner := ⟨.program ⟨214⟩, ⟨26524⟩⟩
def transferEvent : Nat := 106797
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106795 .coefficient) (.predecessor 1 106796 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106795 .coefficient)
      LeftBound101564.bound (LeftBound101564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101564.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106796 .coefficient)
      LeftAuthority106793.bound (LeftAuthority106793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106793.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106793.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101564.bound LeftAuthority106793.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101564.bound, LeftAuthority106793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101564.actual selector witness) * (LeftAuthority106793.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106797

namespace LeftBound106798
def owner : Owner := ⟨.program ⟨214⟩, ⟨26524⟩⟩
def transferEvent : Nat := 106798
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26522⟩⟩]⟩ [⟨.result 106794 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106794 .coefficient)
      LeftAuthority106793.bound (LeftAuthority106793.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26522⟩⟩) (rawTerms := some (Proof.Events417.exact106794RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106793.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106793.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106793.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106793.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106793.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106798

namespace LeftBound106799
def owner : Owner := ⟨.program ⟨214⟩, ⟨26524⟩⟩
def transferEvent : Nat := 106799
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101568 .summary) (.transfer 106798) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101568 .summary)
      LeftBound101567.bound (LeftBound101567.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24977⟩⟩) (rawTerms := some (Proof.Events396.exact101568RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106798)
      LeftBound106798.bound (LeftBound106798.actual selector witness) := by
  exact .transfer (LeftBound106798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101567.bound LeftBound106798.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101567.bound, LeftBound106798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101567.actual selector witness) * (LeftBound106798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106799

namespace LeftBound106810
def owner : Owner := ⟨.program ⟨214⟩, ⟨20455⟩⟩
def transferEvent : Nat := 106810
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106808 .coefficient) (.value (.predecessor 1 106809 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106808 .coefficient)
      LeftAuthority106806.bound (LeftAuthority106806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106809 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106806.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106806.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106806.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106810

namespace LeftBound106814
def owner : Owner := ⟨.program ⟨214⟩, ⟨20456⟩⟩
def transferEvent : Nat := 106814
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106812 .coefficient) (.predecessor 1 106813 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106812 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106813 .coefficient)
      LeftBound106810.bound (LeftBound106810.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106810.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106810.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106810.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106810.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106814

namespace LeftBound106815
def owner : Owner := ⟨.program ⟨214⟩, ⟨20456⟩⟩
def transferEvent : Nat := 106815
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20453⟩⟩]⟩ [⟨.result 106807 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106807 .coefficient)
      LeftAuthority106806.bound (LeftAuthority106806.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20453⟩⟩) (rawTerms := some (Proof.Events417.exact106807RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106806.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106806.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106806.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106815

namespace LeftBound106816
def owner : Owner := ⟨.program ⟨214⟩, ⟨20456⟩⟩
def transferEvent : Nat := 106816
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 106815) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106815)
      LeftBound106815.bound (LeftBound106815.actual selector witness) := by
  exact .transfer (LeftBound106815.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound106815.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound106815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound106815.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106816

namespace LeftBound106887
def owner : Owner := ⟨.program ⟨214⟩, ⟨14944⟩⟩
def transferEvent : Nat := 106887
def frameStart : Nat := 106860
def rule : BoundRule := .identity (.predecessor 0 106886 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106886 .coefficient)
      LeftAuthority106884.bound (LeftAuthority106884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106884.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106884.derived selector witness)

def rawBound : CoeffClass := LeftAuthority106884.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority106884.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106887

namespace LeftBound106904
def owner : Owner := ⟨.program ⟨214⟩, ⟨14985⟩⟩
def transferEvent : Nat := 106904
def frameStart : Nat := 106860
def rule : BoundRule := .sum [.predecessor 0 106902 .coefficient, .predecessor 1 106903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106902 .coefficient)
      LeftBound106887.bound (LeftBound106887.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106903 .coefficient)
      LeftAuthority106900.bound (LeftAuthority106900.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority106900.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106887.bound, LeftAuthority106900.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106887.bound, LeftAuthority106900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106887.actual selector witness, LeftAuthority106900.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106904

namespace LeftBound106907
def owner : Owner := ⟨.program ⟨214⟩, ⟨14986⟩⟩
def transferEvent : Nat := 106907
def frameStart : Nat := 106860
def rule : BoundRule := .identity (.predecessor 0 106906 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106906 .coefficient)
      LeftBound106904.bound (LeftBound106904.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound106904.derived selector witness)

def rawBound : CoeffClass := LeftBound106904.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound106904.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound106907

namespace LeftBound106913
def owner : Owner := ⟨.program ⟨214⟩, ⟨14987⟩⟩
def transferEvent : Nat := 106913
def frameStart : Nat := 106860
def rule : BoundRule := .product (.predecessor 0 106911 .coefficient) (.predecessor 1 106912 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106911 .coefficient)
      LeftAuthority106909.bound (LeftAuthority106909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106912 .coefficient)
      LeftBound106907.bound (LeftBound106907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106907.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority106909.bound LeftBound106907.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106909.bound, LeftBound106907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority106909.actual selector witness) * (LeftBound106907.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106913

namespace LeftBound106921
def owner : Owner := ⟨.program ⟨214⟩, ⟨14988⟩⟩
def transferEvent : Nat := 106921
def frameStart : Nat := 106860
def rule : BoundRule := .sum [.predecessor 0 106919 .coefficient, .predecessor 1 106920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106919 .coefficient)
      LeftAuthority106917.bound (LeftAuthority106917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106920 .coefficient)
      LeftBound106913.bound (LeftBound106913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106913.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106917.bound, LeftBound106913.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106917.bound, LeftBound106913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106917.actual selector witness, LeftBound106913.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106921

namespace LeftBound106925
def owner : Owner := ⟨.program ⟨214⟩, ⟨26523⟩⟩
def transferEvent : Nat := 106925
def frameStart : Nat := 106860
def rule : BoundRule := .product (.predecessor 0 106923 .coefficient) (.predecessor 1 106924 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106923 .coefficient)
      LeftBound106921.bound (LeftBound106921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106924 .coefficient)
      LeftAuthority106898.bound (LeftAuthority106898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106898.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106898.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106921.bound LeftAuthority106898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106921.bound, LeftAuthority106898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106921.actual selector witness) * (LeftAuthority106898.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106925

namespace LeftBound106936
def owner : Owner := ⟨.program ⟨214⟩, ⟨15037⟩⟩
def transferEvent : Nat := 106936
def frameStart : Nat := 106860
def rule : BoundRule := .product (.predecessor 0 106934 .coefficient) (.predecessor 1 106935 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106934 .coefficient)
      LeftAuthority106909.bound (LeftAuthority106909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106909.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106935 .coefficient)
      LeftAuthority106932.bound (LeftAuthority106932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority106909.bound LeftAuthority106932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106909.bound, LeftAuthority106932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority106909.actual selector witness) * (LeftAuthority106932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106936

namespace LeftBound106944
def owner : Owner := ⟨.program ⟨214⟩, ⟨15038⟩⟩
def transferEvent : Nat := 106944
def frameStart : Nat := 106860
def rule : BoundRule := .sum [.predecessor 0 106942 .coefficient, .predecessor 1 106943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106942 .coefficient)
      LeftAuthority106940.bound (LeftAuthority106940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106943 .coefficient)
      LeftBound106936.bound (LeftBound106936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106936.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority106940.bound, LeftBound106936.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106940.bound, LeftBound106936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority106940.actual selector witness, LeftBound106936.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106944

namespace LeftBound106948
def owner : Owner := ⟨.program ⟨214⟩, ⟨26528⟩⟩
def transferEvent : Nat := 106948
def frameStart : Nat := 106860
def rule : BoundRule := .sum [.predecessor 0 106946 .coefficient, .predecessor 1 106947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106946 .coefficient)
      LeftBound106944.bound (LeftBound106944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106947 .coefficient)
      LeftBound106925.bound (LeftBound106925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106925.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106944.bound, LeftBound106925.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106944.bound, LeftBound106925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106944.actual selector witness, LeftBound106925.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106948

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
