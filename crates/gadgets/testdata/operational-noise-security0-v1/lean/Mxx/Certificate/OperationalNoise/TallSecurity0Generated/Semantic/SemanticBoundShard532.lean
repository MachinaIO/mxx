import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard500
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard531

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78979
def owner : Owner := ⟨.program ⟨214⟩, ⟨26765⟩⟩
def transferEvent : Nat := 78979
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 78974 .summary) (.transfer 78978) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78974 .summary)
      LeftBound78973.bound (LeftBound78973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26764⟩⟩) (rawTerms := some (Proof.Events308.exact78974RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78978)
      LeftBound78978.bound (LeftBound78978.actual selector witness) := by
  exact .transfer (LeftBound78978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound78973.bound LeftBound78978.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78973.bound, LeftBound78978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound78973.actual selector witness) * (LeftBound78978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78979

namespace LeftBound78994
def owner : Owner := ⟨.program ⟨214⟩, ⟨26546⟩⟩
def transferEvent : Nat := 78994
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78992 .coefficient) (.predecessor 1 78993 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 78992 .coefficient)
      LeftBound73281.bound (LeftBound73281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events286.exact73285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound73281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound73281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 78993 .coefficient)
      LeftAuthority78990.bound (LeftAuthority78990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact78991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78990.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73281.bound LeftAuthority78990.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73281.bound, LeftAuthority78990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73281.actual selector witness) * (LeftAuthority78990.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78994

namespace LeftBound78995
def owner : Owner := ⟨.program ⟨214⟩, ⟨26546⟩⟩
def transferEvent : Nat := 78995
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26544⟩⟩]⟩ [⟨.result 78991 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 78991 .coefficient)
      LeftAuthority78990.bound (LeftAuthority78990.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26544⟩⟩) (rawTerms := some (Proof.Events308.exact78991RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78990.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78990.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority78990.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority78990.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78995

namespace LeftBound78996
def owner : Owner := ⟨.program ⟨214⟩, ⟨26546⟩⟩
def transferEvent : Nat := 78996
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 73285 .summary) (.transfer 78995) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 73285 .summary)
      LeftBound73284.bound (LeftBound73284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24985⟩⟩) (rawTerms := some (Proof.Events286.exact73285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound73284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 78995)
      LeftBound78995.bound (LeftBound78995.actual selector witness) := by
  exact .transfer (LeftBound78995.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound73284.bound LeftBound78995.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound73284.bound, LeftBound78995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound73284.actual selector witness) * (LeftBound78995.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78996

namespace LeftBound79007
def owner : Owner := ⟨.program ⟨214⟩, ⟨20462⟩⟩
def transferEvent : Nat := 79007
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 79005 .coefficient) (.value (.predecessor 1 79006 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79005 .coefficient)
      LeftAuthority79003.bound (LeftAuthority79003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact79004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79006 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority79003.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79003.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79003.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79007

namespace LeftBound79011
def owner : Owner := ⟨.program ⟨214⟩, ⟨20463⟩⟩
def transferEvent : Nat := 79011
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79009 .coefficient) (.predecessor 1 79010 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79009 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79010 .coefficient)
      LeftBound79007.bound (LeftBound79007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact79008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79007.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound79007.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound79007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound79007.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79011

namespace LeftBound79012
def owner : Owner := ⟨.program ⟨214⟩, ⟨20463⟩⟩
def transferEvent : Nat := 79012
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20460⟩⟩]⟩ [⟨.result 79004 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79004 .coefficient)
      LeftAuthority79003.bound (LeftAuthority79003.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20460⟩⟩) (rawTerms := some (Proof.Events308.exact79004RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79003.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79003.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79003.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79012

namespace LeftBound79013
def owner : Owner := ⟨.program ⟨214⟩, ⟨20463⟩⟩
def transferEvent : Nat := 79013
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 79012) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79012)
      LeftBound79012.bound (LeftBound79012.actual selector witness) := by
  exact .transfer (LeftBound79012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound79012.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound79012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound79012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79013

namespace LeftBound79108
def owner : Owner := ⟨.program ⟨214⟩, ⟨14950⟩⟩
def transferEvent : Nat := 79108
def frameStart : Nat := 79069
def rule : BoundRule := .identity (.predecessor 0 79107 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79107 .coefficient)
      LeftAuthority79105.bound (LeftAuthority79105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79105.derived selector witness)

def rawBound : CoeffClass := LeftAuthority79105.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority79105.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79108

namespace LeftBound79125
def owner : Owner := ⟨.program ⟨214⟩, ⟨14989⟩⟩
def transferEvent : Nat := 79125
def frameStart : Nat := 79069
def rule : BoundRule := .sum [.predecessor 0 79123 .coefficient, .predecessor 1 79124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79123 .coefficient)
      LeftBound79108.bound (LeftBound79108.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79124 .coefficient)
      LeftAuthority79121.bound (LeftAuthority79121.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority79121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79108.bound, LeftAuthority79121.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79108.bound, LeftAuthority79121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79108.actual selector witness, LeftAuthority79121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79125

namespace LeftBound79128
def owner : Owner := ⟨.program ⟨214⟩, ⟨14990⟩⟩
def transferEvent : Nat := 79128
def frameStart : Nat := 79069
def rule : BoundRule := .identity (.predecessor 0 79127 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79127 .coefficient)
      LeftBound79125.bound (LeftBound79125.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79125.derived selector witness)

def rawBound : CoeffClass := LeftBound79125.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound79125.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79128

namespace LeftBound79134
def owner : Owner := ⟨.program ⟨214⟩, ⟨14991⟩⟩
def transferEvent : Nat := 79134
def frameStart : Nat := 79069
def rule : BoundRule := .product (.predecessor 0 79132 .coefficient) (.predecessor 1 79133 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79132 .coefficient)
      LeftAuthority79130.bound (LeftAuthority79130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79133 .coefficient)
      LeftBound79128.bound (LeftBound79128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority79130.bound LeftBound79128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79130.bound, LeftBound79128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority79130.actual selector witness) * (LeftBound79128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79134

namespace LeftBound79142
def owner : Owner := ⟨.program ⟨214⟩, ⟨14992⟩⟩
def transferEvent : Nat := 79142
def frameStart : Nat := 79069
def rule : BoundRule := .sum [.predecessor 0 79140 .coefficient, .predecessor 1 79141 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79140 .coefficient)
      LeftAuthority79138.bound (LeftAuthority79138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79141 .coefficient)
      LeftBound79134.bound (LeftBound79134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79138.bound, LeftBound79134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79138.bound, LeftBound79134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority79138.actual selector witness, LeftBound79134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79142

namespace LeftBound79146
def owner : Owner := ⟨.program ⟨214⟩, ⟨26545⟩⟩
def transferEvent : Nat := 79146
def frameStart : Nat := 79069
def rule : BoundRule := .product (.predecessor 0 79144 .coefficient) (.predecessor 1 79145 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79144 .coefficient)
      LeftBound79142.bound (LeftBound79142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79145 .coefficient)
      LeftAuthority79119.bound (LeftAuthority79119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79119.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79142.bound LeftAuthority79119.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79142.bound, LeftAuthority79119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79142.actual selector witness) * (LeftAuthority79119.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79146

namespace LeftBound79157
def owner : Owner := ⟨.program ⟨214⟩, ⟨15045⟩⟩
def transferEvent : Nat := 79157
def frameStart : Nat := 79069
def rule : BoundRule := .product (.predecessor 0 79155 .coefficient) (.predecessor 1 79156 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79155 .coefficient)
      LeftAuthority79130.bound (LeftAuthority79130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79156 .coefficient)
      LeftAuthority79153.bound (LeftAuthority79153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79153.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79153.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority79130.bound LeftAuthority79153.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79130.bound, LeftAuthority79153.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority79130.actual selector witness) * (LeftAuthority79153.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79157

namespace LeftBound79165
def owner : Owner := ⟨.program ⟨214⟩, ⟨15046⟩⟩
def transferEvent : Nat := 79165
def frameStart : Nat := 79069
def rule : BoundRule := .sum [.predecessor 0 79163 .coefficient, .predecessor 1 79164 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79163 .coefficient)
      LeftAuthority79161.bound (LeftAuthority79161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79161.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79164 .coefficient)
      LeftBound79157.bound (LeftBound79157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79161.bound, LeftBound79157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79161.bound, LeftBound79157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority79161.actual selector witness, LeftBound79157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79165

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
