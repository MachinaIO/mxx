import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard649
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard712

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104101
def owner : Owner := ⟨.program ⟨214⟩, ⟨16962⟩⟩
def transferEvent : Nat := 104101
def frameStart : Nat := 104040
def rule : BoundRule := .sum [.predecessor 0 104099 .coefficient, .predecessor 1 104100 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104099 .coefficient)
      LeftAuthority104097.bound (LeftAuthority104097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104100 .coefficient)
      LeftBound104093.bound (LeftBound104093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104093.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104097.bound, LeftBound104093.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104097.bound, LeftBound104093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104097.actual selector witness, LeftBound104093.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104101

namespace LeftBound104105
def owner : Owner := ⟨.program ⟨214⟩, ⟨29778⟩⟩
def transferEvent : Nat := 104105
def frameStart : Nat := 104040
def rule : BoundRule := .product (.predecessor 0 104103 .coefficient) (.predecessor 1 104104 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104103 .coefficient)
      LeftBound104101.bound (LeftBound104101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104104 .coefficient)
      LeftAuthority104078.bound (LeftAuthority104078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104078.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104101.bound LeftAuthority104078.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104101.bound, LeftAuthority104078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104101.actual selector witness) * (LeftAuthority104078.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104105

namespace LeftBound104116
def owner : Owner := ⟨.program ⟨214⟩, ⟨16919⟩⟩
def transferEvent : Nat := 104116
def frameStart : Nat := 104040
def rule : BoundRule := .product (.predecessor 0 104114 .coefficient) (.predecessor 1 104115 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104114 .coefficient)
      LeftAuthority104089.bound (LeftAuthority104089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104115 .coefficient)
      LeftAuthority104112.bound (LeftAuthority104112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104112.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority104089.bound LeftAuthority104112.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104089.bound, LeftAuthority104112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority104089.actual selector witness) * (LeftAuthority104112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104116

namespace LeftBound104124
def owner : Owner := ⟨.program ⟨214⟩, ⟨16920⟩⟩
def transferEvent : Nat := 104124
def frameStart : Nat := 104040
def rule : BoundRule := .sum [.predecessor 0 104122 .coefficient, .predecessor 1 104123 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104122 .coefficient)
      LeftAuthority104120.bound (LeftAuthority104120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104123 .coefficient)
      LeftBound104116.bound (LeftBound104116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104120.bound, LeftBound104116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104120.bound, LeftBound104116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority104120.actual selector witness, LeftBound104116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104124

namespace LeftBound104128
def owner : Owner := ⟨.program ⟨214⟩, ⟨29783⟩⟩
def transferEvent : Nat := 104128
def frameStart : Nat := 104040
def rule : BoundRule := .sum [.predecessor 0 104126 .coefficient, .predecessor 1 104127 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104126 .coefficient)
      LeftBound104124.bound (LeftBound104124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104127 .coefficient)
      LeftBound104105.bound (LeftBound104105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104124.bound, LeftBound104105.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104124.bound, LeftBound104105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound104124.actual selector witness, LeftBound104105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104128

namespace LeftBound104141
def owner : Owner := ⟨.program ⟨214⟩, ⟨29780⟩⟩
def transferEvent : Nat := 104141
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104139 .coefficient, .predecessor 1 104140 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104139 .coefficient)
      LeftBound103994.bound (LeftBound103994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104140 .coefficient)
      LeftBound103977.bound (LeftBound103977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact103984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103994.bound, LeftBound103977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103994.bound, LeftBound103977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103994.actual selector witness, LeftBound103977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104141

namespace LeftBound104144
def owner : Owner := ⟨.program ⟨214⟩, ⟨29780⟩⟩
def transferEvent : Nat := 104144
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104138 .summary, .result 103984 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104138 .summary)
      LeftBound103996.bound (LeftBound103996.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22616⟩⟩) (rawTerms := some (Proof.Events406.exact104138RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 103984 .summary)
      LeftBound103979.bound (LeftBound103979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29779⟩⟩) (rawTerms := some (Proof.Events406.exact103984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound103996.bound, LeftBound103979.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound103996.bound, LeftBound103979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound103996.actual selector witness, LeftBound103979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104144

namespace LeftBound104148
def owner : Owner := ⟨.program ⟨214⟩, ⟨29781⟩⟩
def transferEvent : Nat := 104148
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104146 .coefficient) (.predecessor 1 104147 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104146 .coefficient)
      LeftBound104141.bound (LeftBound104141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104147 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104141.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104141.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104141.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104148

namespace LeftBound104149
def owner : Owner := ⟨.program ⟨214⟩, ⟨29781⟩⟩
def transferEvent : Nat := 104149
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104149

namespace LeftBound104150
def owner : Owner := ⟨.program ⟨214⟩, ⟨29781⟩⟩
def transferEvent : Nat := 104150
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104145 .summary) (.transfer 104149) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104145 .summary)
      LeftBound104144.bound (LeftBound104144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29780⟩⟩) (rawTerms := some (Proof.Events406.exact104145RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104149)
      LeftBound104149.bound (LeftBound104149.actual selector witness) := by
  exact .transfer (LeftBound104149.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound104144.bound LeftBound104149.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104144.bound, LeftBound104149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound104144.actual selector witness) * (LeftBound104149.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104150

namespace LeftBound104165
def owner : Owner := ⟨.program ⟨214⟩, ⟨29562⟩⟩
def transferEvent : Nat := 104165
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104163 .coefficient) (.predecessor 1 104164 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104163 .coefficient)
      LeftBound95488.bound (LeftBound95488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events373.exact95492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104164 .coefficient)
      LeftAuthority104161.bound (LeftAuthority104161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104161.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95488.bound LeftAuthority104161.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95488.bound, LeftAuthority104161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95488.actual selector witness) * (LeftAuthority104161.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104165

namespace LeftBound104166
def owner : Owner := ⟨.program ⟨214⟩, ⟨29562⟩⟩
def transferEvent : Nat := 104166
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29560⟩⟩]⟩ [⟨.result 104162 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104162 .coefficient)
      LeftAuthority104161.bound (LeftAuthority104161.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29560⟩⟩) (rawTerms := some (Proof.Events406.exact104162RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104161.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104161.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104161.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104166

namespace LeftBound104167
def owner : Owner := ⟨.program ⟨214⟩, ⟨29562⟩⟩
def transferEvent : Nat := 104167
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95492 .summary) (.transfer 104166) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95492 .summary)
      LeftBound95491.bound (LeftBound95491.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25593⟩⟩) (rawTerms := some (Proof.Events373.exact95492RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 104166)
      LeftBound104166.bound (LeftBound104166.actual selector witness) := by
  exact .transfer (LeftBound104166.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95491.bound LeftBound104166.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95491.bound, LeftBound104166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95491.actual selector witness) * (LeftBound104166.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104167

namespace LeftBound104178
def owner : Owner := ⟨.program ⟨214⟩, ⟨22471⟩⟩
def transferEvent : Nat := 104178
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 104176 .coefficient) (.value (.predecessor 1 104177 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104176 .coefficient)
      LeftAuthority104174.bound (LeftAuthority104174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104177 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority104174.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104174.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104174.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104178

namespace LeftBound104182
def owner : Owner := ⟨.program ⟨214⟩, ⟨22472⟩⟩
def transferEvent : Nat := 104182
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104180 .coefficient) (.predecessor 1 104181 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 104180 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 104181 .coefficient)
      LeftBound104178.bound (LeftBound104178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104178.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound104178.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound104178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound104178.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104182

namespace LeftBound104183
def owner : Owner := ⟨.program ⟨214⟩, ⟨22472⟩⟩
def transferEvent : Nat := 104183
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22469⟩⟩]⟩ [⟨.result 104175 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 104175 .coefficient)
      LeftAuthority104174.bound (LeftAuthority104174.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22469⟩⟩) (rawTerms := some (Proof.Events406.exact104175RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104174.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104174.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority104174.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104183

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
