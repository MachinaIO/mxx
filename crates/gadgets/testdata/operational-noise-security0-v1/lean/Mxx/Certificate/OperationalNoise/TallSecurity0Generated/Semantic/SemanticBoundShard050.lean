import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard049

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound9053
def owner : Owner := ⟨.program ⟨214⟩, ⟨25394⟩⟩
def transferEvent : Nat := 9053
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25393⟩⟩]⟩ [⟨.result 8966 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8966 .coefficient)
      LeftAuthority8965.bound (LeftAuthority8965.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25393⟩⟩) (rawTerms := some (Proof.Events035.exact8966RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8965.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8965.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8965.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9053

namespace LeftBound9054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25394⟩⟩
def transferEvent : Nat := 9054
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9049 .summary) (.transfer 9053) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9049 .summary)
      LeftBound9048.bound (LeftBound9048.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12409⟩⟩) (rawTerms := some (Proof.Events035.exact9049RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9053)
      LeftBound9053.bound (LeftBound9053.actual selector witness) := by
  exact .transfer (LeftBound9053.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9048.bound LeftBound9053.bound
def bound : CoeffClass := .finite ⟨350316591579136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9048.bound, LeftBound9053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9048.actual selector witness) * (LeftBound9053.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9054

namespace LeftBound9065
def owner : Owner := ⟨.program ⟨214⟩, ⟨19906⟩⟩
def transferEvent : Nat := 9065
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 9063 .coefficient) (.value (.predecessor 1 9064 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9063 .coefficient)
      LeftAuthority9061.bound (LeftAuthority9061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9061.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9061.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9064 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9061.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9061.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9061.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9065

namespace LeftBound9069
def owner : Owner := ⟨.program ⟨214⟩, ⟨19907⟩⟩
def transferEvent : Nat := 9069
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9067 .coefficient) (.predecessor 1 9068 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9067 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9068 .coefficient)
      LeftBound9065.bound (LeftBound9065.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9066RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9065.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9065.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound9065.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound9065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound9065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9069

namespace LeftBound9070
def owner : Owner := ⟨.program ⟨214⟩, ⟨19907⟩⟩
def transferEvent : Nat := 9070
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19904⟩⟩]⟩ [⟨.result 9062 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9062 .coefficient)
      LeftAuthority9061.bound (LeftAuthority9061.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19904⟩⟩) (rawTerms := some (Proof.Events035.exact9062RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9061.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9061.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9061.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9061.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9070

namespace LeftBound9071
def owner : Owner := ⟨.program ⟨214⟩, ⟨19907⟩⟩
def transferEvent : Nat := 9071
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 9070) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9070)
      LeftBound9070.bound (LeftBound9070.actual selector witness) := by
  exact .transfer (LeftBound9070.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound9070.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound9070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound9070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9071

namespace LeftBound9150
def owner : Owner := ⟨.program ⟨214⟩, ⟨12403⟩⟩
def transferEvent : Nat := 9150
def frameStart : Nat := 9121
def rule : BoundRule := .product (.predecessor 0 9148 .coefficient) (.predecessor 1 9149 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9148 .coefficient)
      LeftAuthority9146.bound (LeftAuthority9146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9146.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9149 .coefficient)
      LeftAuthority9143.bound (LeftAuthority9143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9143.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority9146.bound LeftAuthority9143.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9146.bound, LeftAuthority9143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority9146.actual selector witness) * (LeftAuthority9143.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9150

namespace LeftBound9154
def owner : Owner := ⟨.program ⟨214⟩, ⟨12404⟩⟩
def transferEvent : Nat := 9154
def frameStart : Nat := 9121
def rule : BoundRule := .identity (.predecessor 0 9153 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9153 .coefficient)
      LeftBound9150.bound (LeftBound9150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9150.derived selector witness)

def rawBound : CoeffClass := LeftBound9150.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9150.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9154

namespace LeftBound9171
def owner : Owner := ⟨.program ⟨214⟩, ⟨12482⟩⟩
def transferEvent : Nat := 9171
def frameStart : Nat := 9121
def rule : BoundRule := .sum [.predecessor 0 9169 .coefficient, .predecessor 1 9170 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9169 .coefficient)
      LeftBound9154.bound (LeftBound9154.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9170 .coefficient)
      LeftAuthority9167.bound (LeftAuthority9167.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9154.bound, LeftAuthority9167.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9154.bound, LeftAuthority9167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9154.actual selector witness, LeftAuthority9167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9171

namespace LeftBound9174
def owner : Owner := ⟨.program ⟨214⟩, ⟨12483⟩⟩
def transferEvent : Nat := 9174
def frameStart : Nat := 9121
def rule : BoundRule := .identity (.predecessor 0 9173 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9173 .coefficient)
      LeftBound9171.bound (LeftBound9171.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound9171.derived selector witness)

def rawBound : CoeffClass := LeftBound9171.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound9171.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9174

namespace LeftBound9180
def owner : Owner := ⟨.program ⟨214⟩, ⟨12484⟩⟩
def transferEvent : Nat := 9180
def frameStart : Nat := 9121
def rule : BoundRule := .product (.predecessor 0 9178 .coefficient) (.predecessor 1 9179 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9178 .coefficient)
      LeftAuthority9176.bound (LeftAuthority9176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9179 .coefficient)
      LeftBound9174.bound (LeftBound9174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9174.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority9176.bound LeftBound9174.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9176.bound, LeftBound9174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority9176.actual selector witness) * (LeftBound9174.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9180

namespace LeftBound9196
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 9196
def frameStart : Nat := 9121
def rule : BoundRule := .scale (.predecessor 0 9194 .coefficient) (.value (.predecessor 1 9195 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9194 .coefficient)
      LeftAuthority9192.bound (LeftAuthority9192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9195 .coefficient)
      LeftAuthority9183.bound (LeftAuthority9183.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority9183.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9192.bound LeftAuthority9183.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9192.bound, LeftAuthority9183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9192.actual selector witness) * (LeftAuthority9183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9196

namespace LeftBound9199
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 9199
def frameStart : Nat := 9121
def rule : BoundRule := .identity (.predecessor 0 9198 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9198 .coefficient)
      LeftAuthority9186.bound (LeftAuthority9186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9186.derived selector witness)

def rawBound : CoeffClass := LeftAuthority9186.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority9186.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9199

namespace LeftBound9203
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 9203
def frameStart : Nat := 9121
def rule : BoundRule := .product (.predecessor 0 9201 .coefficient) (.predecessor 1 9202 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9201 .coefficient)
      LeftBound9199.bound (LeftBound9199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9199.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9202 .coefficient)
      LeftBound9196.bound (LeftBound9196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9199.bound LeftBound9196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9199.bound, LeftBound9196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9199.actual selector witness) * (LeftBound9196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9203

namespace LeftBound9208
def owner : Owner := ⟨.program ⟨214⟩, ⟨12485⟩⟩
def transferEvent : Nat := 9208
def frameStart : Nat := 9121
def rule : BoundRule := .sum [.predecessor 0 9206 .coefficient, .predecessor 1 9207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9206 .coefficient)
      LeftBound9203.bound (LeftBound9203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9207 .coefficient)
      LeftBound9180.bound (LeftBound9180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9203.bound, LeftBound9180.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9203.bound, LeftBound9180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9203.actual selector witness, LeftBound9180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9208

namespace LeftBound9212
def owner : Owner := ⟨.program ⟨214⟩, ⟨25396⟩⟩
def transferEvent : Nat := 9212
def frameStart : Nat := 9121
def rule : BoundRule := .product (.predecessor 0 9210 .coefficient) (.predecessor 1 9211 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9210 .coefficient)
      LeftBound9208.bound (LeftBound9208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9211 .coefficient)
      LeftAuthority9165.bound (LeftAuthority9165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9208.bound LeftAuthority9165.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9208.bound, LeftAuthority9165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9208.actual selector witness) * (LeftAuthority9165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9212

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
