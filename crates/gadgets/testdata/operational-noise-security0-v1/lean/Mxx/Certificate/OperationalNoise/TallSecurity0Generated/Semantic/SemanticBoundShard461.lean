import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard460

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67988
def owner : Owner := ⟨.program ⟨214⟩, ⟨28940⟩⟩
def transferEvent : Nat := 67988
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67983 .summary) (.transfer 67987) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67983 .summary)
      LeftBound67982.bound (LeftBound67982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25370⟩⟩) (rawTerms := some (Proof.Events265.exact67983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67987)
      LeftBound67987.bound (LeftBound67987.actual selector witness) := by
  exact .transfer (LeftBound67987.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67982.bound LeftBound67987.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67982.bound, LeftBound67987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67982.actual selector witness) * (LeftBound67987.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67988

namespace LeftBound67999
def owner : Owner := ⟨.program ⟨214⟩, ⟨22118⟩⟩
def transferEvent : Nat := 67999
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 67997 .coefficient) (.value (.predecessor 1 67998 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67997 .coefficient)
      LeftAuthority67995.bound (LeftAuthority67995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67998 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67995.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67995.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67995.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67999

namespace LeftBound68003
def owner : Owner := ⟨.program ⟨214⟩, ⟨22119⟩⟩
def transferEvent : Nat := 68003
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68001 .coefficient) (.predecessor 1 68002 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68001 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68002 .coefficient)
      LeftBound67999.bound (LeftBound67999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact68000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound67999.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound67999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound67999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68003

namespace LeftBound68004
def owner : Owner := ⟨.program ⟨214⟩, ⟨22119⟩⟩
def transferEvent : Nat := 68004
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22116⟩⟩]⟩ [⟨.result 67996 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67996 .coefficient)
      LeftAuthority67995.bound (LeftAuthority67995.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22116⟩⟩) (rawTerms := some (Proof.Events265.exact67996RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67995.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67995.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67995.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67995.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68004

namespace LeftBound68005
def owner : Owner := ⟨.program ⟨214⟩, ⟨22119⟩⟩
def transferEvent : Nat := 68005
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 68004) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68004)
      LeftBound68004.bound (LeftBound68004.actual selector witness) := by
  exact .transfer (LeftBound68004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound68004.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound68004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound68004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68005

namespace LeftBound68100
def owner : Owner := ⟨.program ⟨214⟩, ⟨16462⟩⟩
def transferEvent : Nat := 68100
def frameStart : Nat := 68061
def rule : BoundRule := .identity (.predecessor 0 68099 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68099 .coefficient)
      LeftAuthority68097.bound (LeftAuthority68097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68097.derived selector witness)

def rawBound : CoeffClass := LeftAuthority68097.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority68097.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68100

namespace LeftBound68117
def owner : Owner := ⟨.program ⟨214⟩, ⟨16501⟩⟩
def transferEvent : Nat := 68117
def frameStart : Nat := 68061
def rule : BoundRule := .sum [.predecessor 0 68115 .coefficient, .predecessor 1 68116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68115 .coefficient)
      LeftBound68100.bound (LeftBound68100.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68116 .coefficient)
      LeftAuthority68113.bound (LeftAuthority68113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority68113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68100.bound, LeftAuthority68113.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68100.bound, LeftAuthority68113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68100.actual selector witness, LeftAuthority68113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68117

namespace LeftBound68120
def owner : Owner := ⟨.program ⟨214⟩, ⟨16502⟩⟩
def transferEvent : Nat := 68120
def frameStart : Nat := 68061
def rule : BoundRule := .identity (.predecessor 0 68119 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68119 .coefficient)
      LeftBound68117.bound (LeftBound68117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound68117.derived selector witness)

def rawBound : CoeffClass := LeftBound68117.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound68117.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68120

namespace LeftBound68126
def owner : Owner := ⟨.program ⟨214⟩, ⟨16503⟩⟩
def transferEvent : Nat := 68126
def frameStart : Nat := 68061
def rule : BoundRule := .product (.predecessor 0 68124 .coefficient) (.predecessor 1 68125 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68124 .coefficient)
      LeftAuthority68122.bound (LeftAuthority68122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68125 .coefficient)
      LeftBound68120.bound (LeftBound68120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority68122.bound LeftBound68120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68122.bound, LeftBound68120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority68122.actual selector witness) * (LeftBound68120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68126

namespace LeftBound68134
def owner : Owner := ⟨.program ⟨214⟩, ⟨16504⟩⟩
def transferEvent : Nat := 68134
def frameStart : Nat := 68061
def rule : BoundRule := .sum [.predecessor 0 68132 .coefficient, .predecessor 1 68133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68132 .coefficient)
      LeftAuthority68130.bound (LeftAuthority68130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68133 .coefficient)
      LeftBound68126.bound (LeftBound68126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68130.bound, LeftBound68126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68130.bound, LeftBound68126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68130.actual selector witness, LeftBound68126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68134

namespace LeftBound68138
def owner : Owner := ⟨.program ⟨214⟩, ⟨28939⟩⟩
def transferEvent : Nat := 68138
def frameStart : Nat := 68061
def rule : BoundRule := .product (.predecessor 0 68136 .coefficient) (.predecessor 1 68137 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68136 .coefficient)
      LeftBound68134.bound (LeftBound68134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68137 .coefficient)
      LeftAuthority68111.bound (LeftAuthority68111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68111.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68111.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68134.bound LeftAuthority68111.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68134.bound, LeftAuthority68111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68134.actual selector witness) * (LeftAuthority68111.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68138

namespace LeftBound68149
def owner : Owner := ⟨.program ⟨214⟩, ⟨17902⟩⟩
def transferEvent : Nat := 68149
def frameStart : Nat := 68061
def rule : BoundRule := .product (.predecessor 0 68147 .coefficient) (.predecessor 1 68148 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68147 .coefficient)
      LeftAuthority68122.bound (LeftAuthority68122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68122.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68148 .coefficient)
      LeftAuthority68145.bound (LeftAuthority68145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68145.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68122.bound LeftAuthority68145.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68122.bound, LeftAuthority68145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68122.actual selector witness) * (LeftAuthority68145.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68149

namespace LeftBound68157
def owner : Owner := ⟨.program ⟨214⟩, ⟨17903⟩⟩
def transferEvent : Nat := 68157
def frameStart : Nat := 68061
def rule : BoundRule := .sum [.predecessor 0 68155 .coefficient, .predecessor 1 68156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68155 .coefficient)
      LeftAuthority68153.bound (LeftAuthority68153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68156 .coefficient)
      LeftBound68149.bound (LeftBound68149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68153.bound, LeftBound68149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68153.bound, LeftBound68149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68153.actual selector witness, LeftBound68149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68157

namespace LeftBound68161
def owner : Owner := ⟨.program ⟨214⟩, ⟨28943⟩⟩
def transferEvent : Nat := 68161
def frameStart : Nat := 68061
def rule : BoundRule := .sum [.predecessor 0 68159 .coefficient, .predecessor 1 68160 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68159 .coefficient)
      LeftBound68157.bound (LeftBound68157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68160 .coefficient)
      LeftBound68138.bound (LeftBound68138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68157.bound, LeftBound68138.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68157.bound, LeftBound68138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68157.actual selector witness, LeftBound68138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68161

namespace LeftBound68174
def owner : Owner := ⟨.program ⟨214⟩, ⟨28941⟩⟩
def transferEvent : Nat := 68174
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68172 .coefficient, .predecessor 1 68173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68172 .coefficient)
      LeftBound68003.bound (LeftBound68003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68173 .coefficient)
      LeftBound67986.bound (LeftBound67986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact67993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67986.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68003.bound, LeftBound67986.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68003.bound, LeftBound67986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68003.actual selector witness, LeftBound67986.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68174

namespace LeftBound68177
def owner : Owner := ⟨.program ⟨214⟩, ⟨28941⟩⟩
def transferEvent : Nat := 68177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68171 .summary, .result 67993 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68171 .summary)
      LeftBound68005.bound (LeftBound68005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22119⟩⟩) (rawTerms := some (Proof.Events266.exact68171RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67993 .summary)
      LeftBound67988.bound (LeftBound67988.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28940⟩⟩) (rawTerms := some (Proof.Events265.exact67993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67988.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68005.bound, LeftBound67988.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68005.bound, LeftBound67988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68005.actual selector witness, LeftBound67988.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68177

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
