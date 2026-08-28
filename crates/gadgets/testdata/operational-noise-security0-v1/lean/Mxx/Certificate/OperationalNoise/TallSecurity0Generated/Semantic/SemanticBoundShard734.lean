import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard128
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard733

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound107109
def owner : Owner := ⟨.program ⟨214⟩, ⟨14827⟩⟩
def transferEvent : Nat := 107109
def frameStart : Nat := 107048
def rule : BoundRule := .sum [.predecessor 0 107107 .coefficient, .predecessor 1 107108 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107107 .coefficient)
      LeftAuthority107105.bound (LeftAuthority107105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107108 .coefficient)
      LeftBound107101.bound (LeftBound107101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107101.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority107105.bound, LeftBound107101.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107105.bound, LeftBound107101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority107105.actual selector witness, LeftBound107101.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107109

namespace LeftBound107113
def owner : Owner := ⟨.program ⟨214⟩, ⟨26320⟩⟩
def transferEvent : Nat := 107113
def frameStart : Nat := 107048
def rule : BoundRule := .product (.predecessor 0 107111 .coefficient) (.predecessor 1 107112 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107111 .coefficient)
      LeftBound107109.bound (LeftBound107109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107112 .coefficient)
      LeftAuthority107086.bound (LeftAuthority107086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107086.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107086.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound107109.bound LeftAuthority107086.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107109.bound, LeftAuthority107086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound107109.actual selector witness) * (LeftAuthority107086.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107113

namespace LeftBound107124
def owner : Owner := ⟨.program ⟨214⟩, ⟨14876⟩⟩
def transferEvent : Nat := 107124
def frameStart : Nat := 107048
def rule : BoundRule := .product (.predecessor 0 107122 .coefficient) (.predecessor 1 107123 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107122 .coefficient)
      LeftAuthority107097.bound (LeftAuthority107097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107123 .coefficient)
      LeftAuthority107120.bound (LeftAuthority107120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority107097.bound LeftAuthority107120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107097.bound, LeftAuthority107120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority107097.actual selector witness) * (LeftAuthority107120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107124

namespace LeftBound107132
def owner : Owner := ⟨.program ⟨214⟩, ⟨14877⟩⟩
def transferEvent : Nat := 107132
def frameStart : Nat := 107048
def rule : BoundRule := .sum [.predecessor 0 107130 .coefficient, .predecessor 1 107131 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107130 .coefficient)
      LeftAuthority107128.bound (LeftAuthority107128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107131 .coefficient)
      LeftBound107124.bound (LeftBound107124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority107128.bound, LeftBound107124.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107128.bound, LeftBound107124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority107128.actual selector witness, LeftBound107124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107132

namespace LeftBound107136
def owner : Owner := ⟨.program ⟨214⟩, ⟨26325⟩⟩
def transferEvent : Nat := 107136
def frameStart : Nat := 107048
def rule : BoundRule := .sum [.predecessor 0 107134 .coefficient, .predecessor 1 107135 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107134 .coefficient)
      LeftBound107132.bound (LeftBound107132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107132.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107135 .coefficient)
      LeftBound107113.bound (LeftBound107113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107132.bound, LeftBound107113.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107132.bound, LeftBound107113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107132.actual selector witness, LeftBound107113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107136

namespace LeftBound107149
def owner : Owner := ⟨.program ⟨214⟩, ⟨26322⟩⟩
def transferEvent : Nat := 107149
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107147 .coefficient, .predecessor 1 107148 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107147 .coefficient)
      LeftBound107002.bound (LeftBound107002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107148 .coefficient)
      LeftBound106985.bound (LeftBound106985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106985.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107002.bound, LeftBound106985.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107002.bound, LeftBound106985.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107002.actual selector witness, LeftBound106985.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107149

namespace LeftBound107152
def owner : Owner := ⟨.program ⟨214⟩, ⟨26322⟩⟩
def transferEvent : Nat := 107152
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 107146 .summary, .result 106992 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107146 .summary)
      LeftBound107004.bound (LeftBound107004.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20312⟩⟩) (rawTerms := some (Proof.Events418.exact107146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106992 .summary)
      LeftBound106987.bound (LeftBound106987.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26321⟩⟩) (rawTerms := some (Proof.Events417.exact106992RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107004.bound, LeftBound106987.bound]
def bound : CoeffClass := .finite ⟨1291889174379421642752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107004.bound, LeftBound106987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107004.actual selector witness, LeftBound106987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107152

namespace LeftBound107156
def owner : Owner := ⟨.program ⟨214⟩, ⟨26323⟩⟩
def transferEvent : Nat := 107156
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107154 .coefficient) (.predecessor 1 107155 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107154 .coefficient)
      LeftBound107149.bound (LeftBound107149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107155 .coefficient)
      LeftBound5858.bound (LeftBound5858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound107149.bound LeftBound5858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107149.bound, LeftBound5858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound107149.actual selector witness) * (LeftBound5858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107156

namespace LeftBound107157
def owner : Owner := ⟨.program ⟨214⟩, ⟨26323⟩⟩
def transferEvent : Nat := 107157
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
end LeftBound107157

namespace LeftBound107158
def owner : Owner := ⟨.program ⟨214⟩, ⟨26323⟩⟩
def transferEvent : Nat := 107158
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 107153 .summary) (.transfer 107157) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 107153 .summary)
      LeftBound107152.bound (LeftBound107152.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26322⟩⟩) (rawTerms := some (Proof.Events418.exact107153RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound107152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 107157)
      LeftBound107157.bound (LeftBound107157.actual selector witness) := by
  exact .transfer (LeftBound107157.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound107152.bound LeftBound107157.bound
def bound : CoeffClass := .finite ⟨4741253940199267499646124032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107152.bound, LeftBound107157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound107152.actual selector witness) * (LeftBound107157.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107158

namespace LeftBound107166
def owner : Owner := ⟨.program ⟨214⟩, ⟨6623⟩⟩
def transferEvent : Nat := 107166
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 107164 .coefficient) (.predecessor 1 107165 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107164 .coefficient)
      LeftAuthority722.bound (LeftAuthority722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority722.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority722.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107165 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority722.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority722.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority722.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound107166

namespace LeftBound107171
def owner : Owner := ⟨.program ⟨214⟩, ⟨7097⟩⟩
def transferEvent : Nat := 107171
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107169 .coefficient) (.predecessor 1 107170 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107169 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107170 .coefficient)
      LeftBound5872.bound (LeftBound5872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5872.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound5872.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound5872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound5872.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107171

namespace LeftBound107176
def owner : Owner := ⟨.program ⟨214⟩, ⟨7731⟩⟩
def transferEvent : Nat := 107176
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107174 .coefficient, .predecessor 1 107175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107174 .coefficient)
      LeftBound107171.bound (LeftBound107171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107171.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107175 .coefficient)
      LeftBound107166.bound (LeftBound107166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107171.bound, LeftBound107166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107171.bound, LeftBound107166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107171.actual selector witness, LeftBound107166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107176

namespace LeftBound107180
def owner : Owner := ⟨.program ⟨214⟩, ⟨7732⟩⟩
def transferEvent : Nat := 107180
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107178 .coefficient, .predecessor 1 107179 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107178 .coefficient)
      LeftBound107176.bound (LeftBound107176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107179 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107176.bound, LeftBound20907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107176.bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107176.actual selector witness, LeftBound20907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107180

namespace LeftBound107181
def owner : Owner := ⟨.program ⟨214⟩, ⟨7732⟩⟩
def transferEvent : Nat := 107181
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨74⟩⟩]⟩ [⟨.result 20908 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20908 .coefficient)
      LeftBound20907.bound (LeftBound20907.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨74⟩⟩) (rawTerms := some (Proof.Events081.exact20908RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20907.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20907.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound20907.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107181

namespace LeftBound107186
def owner : Owner := ⟨.program ⟨214⟩, ⟨7805⟩⟩
def transferEvent : Nat := 107186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 107184 .coefficient, .predecessor 1 107185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107184 .coefficient)
      LeftBound107180.bound (LeftBound107180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107185 .coefficient)
      LeftBound107180.bound (LeftBound107180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107180.bound, LeftBound107180.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107180.bound, LeftBound107180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107180.actual selector witness, LeftBound107180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107186

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
