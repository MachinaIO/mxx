import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard483

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71108
def owner : Owner := ⟨.program ⟨214⟩, ⟨11300⟩⟩
def transferEvent : Nat := 71108
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩ [⟨.result 12476 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12476 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨91⟩⟩) (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12475.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12475.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71108

namespace LeftBound71113
def owner : Owner := ⟨.program ⟨214⟩, ⟨13767⟩⟩
def transferEvent : Nat := 71113
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71111 .coefficient) (.predecessor 1 71112 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71111 .coefficient)
      LeftBound71107.bound (LeftBound71107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71112 .coefficient)
      LeftAuthority3364.bound (LeftAuthority3364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3364.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound71107.bound LeftAuthority3364.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71107.bound, LeftAuthority3364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound71107.actual selector witness) * (LeftAuthority3364.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71113

namespace LeftBound71114
def owner : Owner := ⟨.program ⟨214⟩, ⟨13767⟩⟩
def transferEvent : Nat := 71114
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13764⟩⟩], []⟩ [⟨.result 3365 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3365 .coefficient)
      LeftAuthority3364.bound (LeftAuthority3364.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13764⟩⟩) (rawTerms := some (Proof.Events013.exact3365RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3364.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3364.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3364.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71114

namespace LeftBound71115
def owner : Owner := ⟨.program ⟨214⟩, ⟨13767⟩⟩
def transferEvent : Nat := 71115
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71110 .summary) (.transfer 71114) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71110 .summary)
      LeftBound71108.bound (LeftBound71108.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11300⟩⟩) (rawTerms := some (Proof.Events277.exact71110RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71114)
      LeftBound71114.bound (LeftBound71114.actual selector witness) := by
  exact .transfer (LeftBound71114.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound71108.bound LeftBound71114.bound
def bound : CoeffClass := .finite ⟨9984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71108.bound, LeftBound71114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound71108.actual selector witness) * (LeftBound71114.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71115

namespace LeftBound71121
def owner : Owner := ⟨.program ⟨214⟩, ⟨13768⟩⟩
def transferEvent : Nat := 71121
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 71119 .coefficient) (.predecessor 1 71120 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71119 .coefficient)
      LeftAuthority3364.bound (LeftAuthority3364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3364.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3364.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71120 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3364.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3364.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3364.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71121

namespace LeftBound71126
def owner : Owner := ⟨.program ⟨214⟩, ⟨7212⟩⟩
def transferEvent : Nat := 71126
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71124 .coefficient) (.predecessor 1 71125 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71124 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71125 .coefficient)
      LeftBound12524.bound (LeftBound12524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound12524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound12524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound12524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71126

namespace LeftBound71131
def owner : Owner := ⟨.program ⟨214⟩, ⟨13769⟩⟩
def transferEvent : Nat := 71131
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71129 .coefficient, .predecessor 1 71130 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71129 .coefficient)
      LeftBound71126.bound (LeftBound71126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71130 .coefficient)
      LeftBound71121.bound (LeftBound71121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71126.bound, LeftBound71121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71126.bound, LeftBound71121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71126.actual selector witness, LeftBound71121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71131

namespace LeftBound71135
def owner : Owner := ⟨.program ⟨214⟩, ⟨13770⟩⟩
def transferEvent : Nat := 71135
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71133 .coefficient, .predecessor 1 71134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71133 .coefficient)
      LeftBound71131.bound (LeftBound71131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71134 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71131.bound, LeftBound12516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71131.bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71131.actual selector witness, LeftBound12516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71135

namespace LeftBound71136
def owner : Owner := ⟨.program ⟨214⟩, ⟨13770⟩⟩
def transferEvent : Nat := 71136
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩ [⟨.result 12517 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12517 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨108⟩⟩) (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12516.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12516.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71136

namespace LeftBound71141
def owner : Owner := ⟨.program ⟨214⟩, ⟨13771⟩⟩
def transferEvent : Nat := 71141
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71139 .coefficient) (.predecessor 1 71140 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71139 .coefficient)
      LeftBound71135.bound (LeftBound71135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71140 .coefficient)
      LeftBound12513.bound (LeftBound12513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71135.bound LeftBound12513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71135.bound, LeftBound12513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71135.actual selector witness) * (LeftBound12513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71141

namespace LeftBound71142
def owner : Owner := ⟨.program ⟨214⟩, ⟨13771⟩⟩
def transferEvent : Nat := 71142
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩ [⟨.result 12510 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12510 .coefficient)
      LeftAuthority12509.bound (LeftAuthority12509.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7846⟩⟩) (rawTerms := some (Proof.Events048.exact12510RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12509.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12509.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12509.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71142

namespace LeftBound71143
def owner : Owner := ⟨.program ⟨214⟩, ⟨13771⟩⟩
def transferEvent : Nat := 71143
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71138 .summary) (.transfer 71142) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71138 .summary)
      LeftBound71136.bound (LeftBound71136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13770⟩⟩) (rawTerms := some (Proof.Events277.exact71138RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71142)
      LeftBound71142.bound (LeftBound71142.actual selector witness) := by
  exact .transfer (LeftBound71142.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71136.bound LeftBound71142.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71136.bound, LeftBound71142.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71136.actual selector witness) * (LeftBound71142.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71143

namespace LeftBound71151
def owner : Owner := ⟨.program ⟨214⟩, ⟨13772⟩⟩
def transferEvent : Nat := 71151
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 71149 .coefficient, .predecessor 1 71150 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71149 .coefficient)
      LeftBound71141.bound (LeftBound71141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71150 .coefficient)
      LeftBound71113.bound (LeftBound71113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71141.bound, LeftBound71113.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71141.bound, LeftBound71113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71141.actual selector witness, LeftBound71113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71151

namespace LeftBound71153
def owner : Owner := ⟨.program ⟨214⟩, ⟨13772⟩⟩
def transferEvent : Nat := 71153
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 71148 .summary, .result 71118 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71148 .summary)
      LeftBound71143.bound (LeftBound71143.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13771⟩⟩) (rawTerms := some (Proof.Events277.exact71148RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71118 .summary)
      LeftBound71115.bound (LeftBound71115.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13767⟩⟩) (rawTerms := some (Proof.Events277.exact71118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71143.bound, LeftBound71115.bound]
def bound : CoeffClass := .finite ⟨95430400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71143.bound, LeftBound71115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71143.actual selector witness, LeftBound71115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71153

namespace LeftBound71157
def owner : Owner := ⟨.program ⟨214⟩, ⟨25908⟩⟩
def transferEvent : Nat := 71157
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71155 .coefficient) (.predecessor 1 71156 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71155 .coefficient)
      LeftBound71151.bound (LeftBound71151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71156 .coefficient)
      LeftAuthority71089.bound (LeftAuthority71089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events277.exact71090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71151.bound LeftAuthority71089.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71151.bound, LeftAuthority71089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71151.actual selector witness) * (LeftAuthority71089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71157

namespace LeftBound71158
def owner : Owner := ⟨.program ⟨214⟩, ⟨25908⟩⟩
def transferEvent : Nat := 71158
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25907⟩⟩]⟩ [⟨.result 71090 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71090 .coefficient)
      LeftAuthority71089.bound (LeftAuthority71089.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25907⟩⟩) (rawTerms := some (Proof.Events277.exact71090RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71089.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71089.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71089.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71158

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
