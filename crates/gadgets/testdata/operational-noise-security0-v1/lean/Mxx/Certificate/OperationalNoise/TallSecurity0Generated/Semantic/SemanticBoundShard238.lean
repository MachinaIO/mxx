import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard234
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard237

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36096
def owner : Owner := ⟨.program ⟨214⟩, ⟨10359⟩⟩
def transferEvent : Nat := 36096
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36094 .coefficient) (.predecessor 1 36095 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36094 .coefficient)
      LeftBound36090.bound (LeftBound36090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36095 .coefficient)
      LeftBound6486.bound (LeftBound6486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36090.bound LeftBound6486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36090.bound, LeftBound6486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36090.actual selector witness) * (LeftBound6486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36096

namespace LeftBound36097
def owner : Owner := ⟨.program ⟨214⟩, ⟨10359⟩⟩
def transferEvent : Nat := 36097
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩ [⟨.result 6483 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6483 .coefficient)
      LeftAuthority6482.bound (LeftAuthority6482.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7882⟩⟩) (rawTerms := some (Proof.Events025.exact6483RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6482.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6482.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6482.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36097

namespace LeftBound36098
def owner : Owner := ⟨.program ⟨214⟩, ⟨10359⟩⟩
def transferEvent : Nat := 36098
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36093 .summary) (.transfer 36097) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36093 .summary)
      LeftBound36091.bound (LeftBound36091.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10358⟩⟩) (rawTerms := some (Proof.Events140.exact36093RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36097)
      LeftBound36097.bound (LeftBound36097.actual selector witness) := by
  exact .transfer (LeftBound36097.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36091.bound LeftBound36097.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36091.bound, LeftBound36097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36091.actual selector witness) * (LeftBound36097.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36098

namespace LeftBound36106
def owner : Owner := ⟨.program ⟨214⟩, ⟨13373⟩⟩
def transferEvent : Nat := 36106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36104 .coefficient, .predecessor 1 36105 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36104 .coefficient)
      LeftBound36096.bound (LeftBound36096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36105 .coefficient)
      LeftBound36068.bound (LeftBound36068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36096.bound, LeftBound36068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36096.bound, LeftBound36068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36096.actual selector witness, LeftBound36068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36106

namespace LeftBound36108
def owner : Owner := ⟨.program ⟨214⟩, ⟨13373⟩⟩
def transferEvent : Nat := 36108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36103 .summary, .result 36073 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36103 .summary)
      LeftBound36098.bound (LeftBound36098.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10359⟩⟩) (rawTerms := some (Proof.Events141.exact36103RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36073 .summary)
      LeftBound36070.bound (LeftBound36070.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13372⟩⟩) (rawTerms := some (Proof.Events140.exact36073RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36098.bound, LeftBound36070.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36098.bound, LeftBound36070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36098.actual selector witness, LeftBound36070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36108

namespace LeftBound36112
def owner : Owner := ⟨.program ⟨214⟩, ⟨25769⟩⟩
def transferEvent : Nat := 36112
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36110 .coefficient) (.predecessor 1 36111 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36110 .coefficient)
      LeftBound36106.bound (LeftBound36106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36111 .coefficient)
      LeftAuthority36039.bound (LeftAuthority36039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact36040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36106.bound LeftAuthority36039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36106.bound, LeftAuthority36039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36106.actual selector witness) * (LeftAuthority36039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36112

namespace LeftBound36113
def owner : Owner := ⟨.program ⟨214⟩, ⟨25769⟩⟩
def transferEvent : Nat := 36113
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25768⟩⟩]⟩ [⟨.result 36040 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36040 .coefficient)
      LeftAuthority36039.bound (LeftAuthority36039.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25768⟩⟩) (rawTerms := some (Proof.Events140.exact36040RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36039.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36039.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36039.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36113

namespace LeftBound36114
def owner : Owner := ⟨.program ⟨214⟩, ⟨25769⟩⟩
def transferEvent : Nat := 36114
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36109 .summary) (.transfer 36113) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36109 .summary)
      LeftBound36108.bound (LeftBound36108.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13373⟩⟩) (rawTerms := some (Proof.Events141.exact36109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36113)
      LeftBound36113.bound (LeftBound36113.actual selector witness) := by
  exact .transfer (LeftBound36113.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36108.bound LeftBound36113.bound
def bound : CoeffClass := .finite ⟨350377660645376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36108.bound, LeftBound36113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36108.actual selector witness) * (LeftBound36113.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36114

namespace LeftBound36125
def owner : Owner := ⟨.program ⟨214⟩, ⟨20258⟩⟩
def transferEvent : Nat := 36125
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 36123 .coefficient) (.value (.predecessor 1 36124 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36123 .coefficient)
      LeftAuthority36121.bound (LeftAuthority36121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36124 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36121.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36121.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36121.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36125

namespace LeftBound36129
def owner : Owner := ⟨.program ⟨214⟩, ⟨5552⟩⟩
def transferEvent : Nat := 36129
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36127 .coefficient) (.predecessor 1 36128 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36127 .coefficient)
      LeftBound35914.bound (LeftBound35914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events140.exact35915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36128 .coefficient)
      LeftAuthority6549.bound (LeftAuthority6549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound35914.bound LeftAuthority6549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35914.bound, LeftAuthority6549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound35914.actual selector witness) * (LeftAuthority6549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 16) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36129

namespace LeftBound36134
def owner : Owner := ⟨.program ⟨214⟩, ⟨5553⟩⟩
def transferEvent : Nat := 36134
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36132 .coefficient, .predecessor 1 36133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36132 .coefficient)
      LeftBound36129.bound (LeftBound36129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36133 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36129.bound, LeftAuthority6547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36129.bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound36129.actual selector witness, LeftAuthority6547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36134

namespace LeftBound36135
def owner : Owner := ⟨.program ⟨214⟩, ⟨5553⟩⟩
def transferEvent : Nat := 36135
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩ [⟨.result 6548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6548 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22⟩⟩) (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6547.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36135

namespace LeftBound36140
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def transferEvent : Nat := 36140
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36138 .coefficient) (.predecessor 1 36139 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36138 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36139 .coefficient)
      LeftBound36125.bound (LeftBound36125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound36125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound36125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound36125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36140

namespace LeftBound36141
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def transferEvent : Nat := 36141
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20256⟩⟩]⟩ [⟨.result 36122 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36122 .coefficient)
      LeftAuthority36121.bound (LeftAuthority36121.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20256⟩⟩) (rawTerms := some (Proof.Events141.exact36122RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36121.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36121.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority36121.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36141

namespace LeftBound36142
def owner : Owner := ⟨.program ⟨214⟩, ⟨20259⟩⟩
def transferEvent : Nat := 36142
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 36141) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 36141)
      LeftBound36141.bound (LeftBound36141.actual selector witness) := by
  exact .transfer (LeftBound36141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound36141.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound36141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound36141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36142

namespace LeftBound36221
def owner : Owner := ⟨.program ⟨214⟩, ⟨13367⟩⟩
def transferEvent : Nat := 36221
def frameStart : Nat := 36192
def rule : BoundRule := .product (.predecessor 0 36219 .coefficient) (.predecessor 1 36220 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 36219 .coefficient)
      LeftAuthority36217.bound (LeftAuthority36217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 36220 .coefficient)
      LeftAuthority36214.bound (LeftAuthority36214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority36217.bound LeftAuthority36214.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36217.bound, LeftAuthority36214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority36217.actual selector witness) * (LeftAuthority36214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36221

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
