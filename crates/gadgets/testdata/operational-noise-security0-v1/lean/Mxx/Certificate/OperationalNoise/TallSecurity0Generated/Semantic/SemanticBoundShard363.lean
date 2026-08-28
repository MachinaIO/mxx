import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard361
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard362

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53839
def owner : Owner := ⟨.program ⟨214⟩, ⟨25226⟩⟩
def transferEvent : Nat := 53839
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53833 .summary, .result 53647 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53833 .summary)
      LeftBound53659.bound (LeftBound53659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19823⟩⟩) (rawTerms := some (Proof.Events210.exact53833RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53647 .summary)
      LeftBound53642.bound (LeftBound53642.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25225⟩⟩) (rawTerms := some (Proof.Events209.exact53647RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53659.bound, LeftBound53642.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53659.bound, LeftBound53642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53659.actual selector witness, LeftBound53642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53839

namespace LeftBound53843
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def transferEvent : Nat := 53843
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53841 .coefficient) (.predecessor 1 53842 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53841 .coefficient)
      LeftBound53836.bound (LeftBound53836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53842 .coefficient)
      LeftAuthority53562.bound (LeftAuthority53562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53836.bound LeftAuthority53562.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53836.bound, LeftAuthority53562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53836.actual selector witness) * (LeftAuthority53562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53843

namespace LeftBound53844
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def transferEvent : Nat := 53844
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28747⟩⟩]⟩ [⟨.result 53563 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53563 .coefficient)
      LeftAuthority53562.bound (LeftAuthority53562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28747⟩⟩) (rawTerms := some (Proof.Events209.exact53563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53562.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53562.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53562.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53844

namespace LeftBound53845
def owner : Owner := ⟨.program ⟨214⟩, ⟨28749⟩⟩
def transferEvent : Nat := 53845
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53840 .summary) (.transfer 53844) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53840 .summary)
      LeftBound53839.bound (LeftBound53839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25226⟩⟩) (rawTerms := some (Proof.Events210.exact53840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53844)
      LeftBound53844.bound (LeftBound53844.actual selector witness) := by
  exact .transfer (LeftBound53844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53839.bound LeftBound53844.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53839.bound, LeftBound53844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53839.actual selector witness) * (LeftBound53844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53845

namespace LeftBound53856
def owner : Owner := ⟨.program ⟨214⟩, ⟨21982⟩⟩
def transferEvent : Nat := 53856
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 53854 .coefficient) (.value (.predecessor 1 53855 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53854 .coefficient)
      LeftAuthority53852.bound (LeftAuthority53852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53855 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53852.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53852.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53852.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53856

namespace LeftBound53860
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def transferEvent : Nat := 53860
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53858 .coefficient) (.predecessor 1 53859 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53858 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53859 .coefficient)
      LeftBound53856.bound (LeftBound53856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound53856.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound53856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound53856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53860

namespace LeftBound53861
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def transferEvent : Nat := 53861
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21980⟩⟩]⟩ [⟨.result 53853 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53853 .coefficient)
      LeftAuthority53852.bound (LeftAuthority53852.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21980⟩⟩) (rawTerms := some (Proof.Events210.exact53853RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53852.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53852.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53852.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53861

namespace LeftBound53862
def owner : Owner := ⟨.program ⟨214⟩, ⟨21983⟩⟩
def transferEvent : Nat := 53862
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 53861) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53861)
      LeftBound53861.bound (LeftBound53861.actual selector witness) := by
  exact .transfer (LeftBound53861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound53861.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound53861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound53861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53862

namespace LeftBound53957
def owner : Owner := ⟨.program ⟨214⟩, ⟨16386⟩⟩
def transferEvent : Nat := 53957
def frameStart : Nat := 53918
def rule : BoundRule := .identity (.predecessor 0 53956 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53956 .coefficient)
      LeftAuthority53954.bound (LeftAuthority53954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53954.derived selector witness)

def rawBound : CoeffClass := LeftAuthority53954.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority53954.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53957

namespace LeftBound53974
def owner : Owner := ⟨.program ⟨214⟩, ⟨16425⟩⟩
def transferEvent : Nat := 53974
def frameStart : Nat := 53918
def rule : BoundRule := .sum [.predecessor 0 53972 .coefficient, .predecessor 1 53973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53972 .coefficient)
      LeftBound53957.bound (LeftBound53957.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53973 .coefficient)
      LeftAuthority53970.bound (LeftAuthority53970.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53957.bound, LeftAuthority53970.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53957.bound, LeftAuthority53970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53957.actual selector witness, LeftAuthority53970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53974

namespace LeftBound53977
def owner : Owner := ⟨.program ⟨214⟩, ⟨16426⟩⟩
def transferEvent : Nat := 53977
def frameStart : Nat := 53918
def rule : BoundRule := .identity (.predecessor 0 53976 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53976 .coefficient)
      LeftBound53974.bound (LeftBound53974.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53974.derived selector witness)

def rawBound : CoeffClass := LeftBound53974.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53974.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53977

namespace LeftBound53983
def owner : Owner := ⟨.program ⟨214⟩, ⟨16427⟩⟩
def transferEvent : Nat := 53983
def frameStart : Nat := 53918
def rule : BoundRule := .product (.predecessor 0 53981 .coefficient) (.predecessor 1 53982 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53981 .coefficient)
      LeftAuthority53979.bound (LeftAuthority53979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53982 .coefficient)
      LeftBound53977.bound (LeftBound53977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53977.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority53979.bound LeftBound53977.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53979.bound, LeftBound53977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority53979.actual selector witness) * (LeftBound53977.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53983

namespace LeftBound53991
def owner : Owner := ⟨.program ⟨214⟩, ⟨16428⟩⟩
def transferEvent : Nat := 53991
def frameStart : Nat := 53918
def rule : BoundRule := .sum [.predecessor 0 53989 .coefficient, .predecessor 1 53990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53989 .coefficient)
      LeftAuthority53987.bound (LeftAuthority53987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53987.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53990 .coefficient)
      LeftBound53983.bound (LeftBound53983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53983.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53987.bound, LeftBound53983.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53987.bound, LeftBound53983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53987.actual selector witness, LeftBound53983.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53991

namespace LeftBound53995
def owner : Owner := ⟨.program ⟨214⟩, ⟨28748⟩⟩
def transferEvent : Nat := 53995
def frameStart : Nat := 53918
def rule : BoundRule := .product (.predecessor 0 53993 .coefficient) (.predecessor 1 53994 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53993 .coefficient)
      LeftBound53991.bound (LeftBound53991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53994 .coefficient)
      LeftAuthority53968.bound (LeftAuthority53968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53991.bound LeftAuthority53968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53991.bound, LeftAuthority53968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53991.actual selector witness) * (LeftAuthority53968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53995

namespace LeftBound54006
def owner : Owner := ⟨.program ⟨214⟩, ⟨17124⟩⟩
def transferEvent : Nat := 54006
def frameStart : Nat := 53918
def rule : BoundRule := .product (.predecessor 0 54004 .coefficient) (.predecessor 1 54005 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54004 .coefficient)
      LeftAuthority53979.bound (LeftAuthority53979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54005 .coefficient)
      LeftAuthority54002.bound (LeftAuthority54002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact54003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54002.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53979.bound LeftAuthority54002.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53979.bound, LeftAuthority54002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53979.actual selector witness) * (LeftAuthority54002.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54006

namespace LeftBound54014
def owner : Owner := ⟨.program ⟨214⟩, ⟨17125⟩⟩
def transferEvent : Nat := 54014
def frameStart : Nat := 53918
def rule : BoundRule := .sum [.predecessor 0 54012 .coefficient, .predecessor 1 54013 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54012 .coefficient)
      LeftAuthority54010.bound (LeftAuthority54010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact54011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54010.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54013 .coefficient)
      LeftBound54006.bound (LeftBound54006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact54008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54010.bound, LeftBound54006.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54010.bound, LeftBound54006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54010.actual selector witness, LeftBound54006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54014

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
