import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard057
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard364

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54106
def owner : Owner := ⟨.program ⟨214⟩, ⟨9619⟩⟩
def transferEvent : Nat := 54106
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54104 .coefficient) (.predecessor 1 54105 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54104 .coefficient)
      LeftBound54100.bound (LeftBound54100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54105 .coefficient)
      LeftBound10008.bound (LeftBound10008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54100.bound LeftBound10008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54100.bound, LeftBound10008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54100.actual selector witness) * (LeftBound10008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54106

namespace LeftBound54107
def owner : Owner := ⟨.program ⟨214⟩, ⟨9619⟩⟩
def transferEvent : Nat := 54107
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩ [⟨.result 10005 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10005 .coefficient)
      LeftAuthority10004.bound (LeftAuthority10004.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7861⟩⟩) (rawTerms := some (Proof.Events039.exact10005RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10004.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10004.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10004.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54107

namespace LeftBound54108
def owner : Owner := ⟨.program ⟨214⟩, ⟨9619⟩⟩
def transferEvent : Nat := 54108
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54103 .summary) (.transfer 54107) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54103 .summary)
      LeftBound54101.bound (LeftBound54101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9618⟩⟩) (rawTerms := some (Proof.Events211.exact54103RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54107)
      LeftBound54107.bound (LeftBound54107.actual selector witness) := by
  exact .transfer (LeftBound54107.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54101.bound LeftBound54107.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54101.bound, LeftBound54107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54101.actual selector witness) * (LeftBound54107.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54108

namespace LeftBound54116
def owner : Owner := ⟨.program ⟨214⟩, ⟨11776⟩⟩
def transferEvent : Nat := 54116
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54114 .coefficient, .predecessor 1 54115 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54114 .coefficient)
      LeftBound54106.bound (LeftBound54106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54115 .coefficient)
      LeftBound54078.bound (LeftBound54078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54106.bound, LeftBound54078.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54106.bound, LeftBound54078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54106.actual selector witness, LeftBound54078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54116

namespace LeftBound54118
def owner : Owner := ⟨.program ⟨214⟩, ⟨11776⟩⟩
def transferEvent : Nat := 54118
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54113 .summary, .result 54083 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54113 .summary)
      LeftBound54108.bound (LeftBound54108.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9619⟩⟩) (rawTerms := some (Proof.Events211.exact54113RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54083 .summary)
      LeftBound54080.bound (LeftBound54080.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11775⟩⟩) (rawTerms := some (Proof.Events211.exact54083RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54080.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54108.bound, LeftBound54080.bound]
def bound : CoeffClass := .finite ⟨95445376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54108.bound, LeftBound54080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54108.actual selector witness, LeftBound54080.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54118

namespace LeftBound54122
def owner : Owner := ⟨.program ⟨214⟩, ⟨25148⟩⟩
def transferEvent : Nat := 54122
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54120 .coefficient) (.predecessor 1 54121 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54120 .coefficient)
      LeftBound54116.bound (LeftBound54116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54121 .coefficient)
      LeftAuthority54054.bound (LeftAuthority54054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54054.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54116.bound LeftAuthority54054.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54116.bound, LeftAuthority54054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54116.actual selector witness) * (LeftAuthority54054.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54122

namespace LeftBound54123
def owner : Owner := ⟨.program ⟨214⟩, ⟨25148⟩⟩
def transferEvent : Nat := 54123
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25147⟩⟩]⟩ [⟨.result 54055 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54055 .coefficient)
      LeftAuthority54054.bound (LeftAuthority54054.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25147⟩⟩) (rawTerms := some (Proof.Events211.exact54055RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54054.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54054.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54054.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54054.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54123

namespace LeftBound54124
def owner : Owner := ⟨.program ⟨214⟩, ⟨25148⟩⟩
def transferEvent : Nat := 54124
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54119 .summary) (.transfer 54123) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54119 .summary)
      LeftBound54118.bound (LeftBound54118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11776⟩⟩) (rawTerms := some (Proof.Events211.exact54119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54123)
      LeftBound54123.bound (LeftBound54123.actual selector witness) := by
  exact .transfer (LeftBound54123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54118.bound LeftBound54123.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54118.bound, LeftBound54123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54118.actual selector witness) * (LeftBound54123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54124

namespace LeftBound54135
def owner : Owner := ⟨.program ⟨214⟩, ⟨19750⟩⟩
def transferEvent : Nat := 54135
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 54133 .coefficient) (.value (.predecessor 1 54134 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54133 .coefficient)
      LeftAuthority54131.bound (LeftAuthority54131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54134 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54131.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54131.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54131.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54135

namespace LeftBound54139
def owner : Owner := ⟨.program ⟨214⟩, ⟨19751⟩⟩
def transferEvent : Nat := 54139
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54137 .coefficient) (.predecessor 1 54138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54137 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54138 .coefficient)
      LeftBound54135.bound (LeftBound54135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54135.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound54135.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound54135.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound54135.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54139

namespace LeftBound54140
def owner : Owner := ⟨.program ⟨214⟩, ⟨19751⟩⟩
def transferEvent : Nat := 54140
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19748⟩⟩]⟩ [⟨.result 54132 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54132 .coefficient)
      LeftAuthority54131.bound (LeftAuthority54131.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19748⟩⟩) (rawTerms := some (Proof.Events211.exact54132RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54131.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54131.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54131.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54140

namespace LeftBound54141
def owner : Owner := ⟨.program ⟨214⟩, ⟨19751⟩⟩
def transferEvent : Nat := 54141
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 54140) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54140)
      LeftBound54140.bound (LeftBound54140.actual selector witness) := by
  exact .transfer (LeftBound54140.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound54140.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound54140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound54140.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54141

namespace LeftBound54220
def owner : Owner := ⟨.program ⟨214⟩, ⟨11770⟩⟩
def transferEvent : Nat := 54220
def frameStart : Nat := 54191
def rule : BoundRule := .product (.predecessor 0 54218 .coefficient) (.predecessor 1 54219 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54218 .coefficient)
      LeftAuthority54216.bound (LeftAuthority54216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54216.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54219 .coefficient)
      LeftAuthority54213.bound (LeftAuthority54213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54213.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54213.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54216.bound LeftAuthority54213.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54216.bound, LeftAuthority54213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority54216.actual selector witness) * (LeftAuthority54213.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54220

namespace LeftBound54224
def owner : Owner := ⟨.program ⟨214⟩, ⟨11771⟩⟩
def transferEvent : Nat := 54224
def frameStart : Nat := 54191
def rule : BoundRule := .identity (.predecessor 0 54223 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54223 .coefficient)
      LeftBound54220.bound (LeftBound54220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54222RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54220.derived selector witness)

def rawBound : CoeffClass := LeftBound54220.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound54220.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54224

namespace LeftBound54241
def owner : Owner := ⟨.program ⟨214⟩, ⟨11861⟩⟩
def transferEvent : Nat := 54241
def frameStart : Nat := 54191
def rule : BoundRule := .sum [.predecessor 0 54239 .coefficient, .predecessor 1 54240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54239 .coefficient)
      LeftBound54224.bound (LeftBound54224.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54240 .coefficient)
      LeftAuthority54237.bound (LeftAuthority54237.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54224.bound, LeftAuthority54237.bound]
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54224.bound, LeftAuthority54237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54224.actual selector witness, LeftAuthority54237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54241

namespace LeftBound54244
def owner : Owner := ⟨.program ⟨214⟩, ⟨11862⟩⟩
def transferEvent : Nat := 54244
def frameStart : Nat := 54191
def rule : BoundRule := .identity (.predecessor 0 54243 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54243 .coefficient)
      LeftBound54241.bound (LeftBound54241.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54241.derived selector witness)

def rawBound : CoeffClass := LeftBound54241.bound
def bound : CoeffClass := .finite ⟨900, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound54241.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54244

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
