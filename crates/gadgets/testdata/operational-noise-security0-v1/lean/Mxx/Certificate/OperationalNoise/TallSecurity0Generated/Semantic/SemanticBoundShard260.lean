import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard259

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound39000
def owner : Owner := ⟨.program ⟨214⟩, ⟨9729⟩⟩
def transferEvent : Nat := 39000
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7864⟩⟩]⟩ [⟨.result 9504 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9504 .coefficient)
      LeftAuthority9503.bound (LeftAuthority9503.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7864⟩⟩) (rawTerms := some (Proof.Events037.exact9504RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9503.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9503.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9503.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39000

namespace LeftBound39001
def owner : Owner := ⟨.program ⟨214⟩, ⟨9729⟩⟩
def transferEvent : Nat := 39001
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 38996 .summary) (.transfer 39000) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38996 .summary)
      LeftBound38994.bound (LeftBound38994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9728⟩⟩) (rawTerms := some (Proof.Events152.exact38996RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39000)
      LeftBound39000.bound (LeftBound39000.actual selector witness) := by
  exact .transfer (LeftBound39000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound38994.bound LeftBound39000.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38994.bound, LeftBound39000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound38994.actual selector witness) * (LeftBound39000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39001

namespace LeftBound39009
def owner : Owner := ⟨.program ⟨214⟩, ⟨11980⟩⟩
def transferEvent : Nat := 39009
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 39007 .coefficient, .predecessor 1 39008 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39007 .coefficient)
      LeftBound38999.bound (LeftBound38999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39008 .coefficient)
      LeftBound38971.bound (LeftBound38971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound38971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound38971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound38999.bound, LeftBound38971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound38999.bound, LeftBound38971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound38999.actual selector witness, LeftBound38971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39009

namespace LeftBound39011
def owner : Owner := ⟨.program ⟨214⟩, ⟨11980⟩⟩
def transferEvent : Nat := 39011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 39006 .summary, .result 38976 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39006 .summary)
      LeftBound39001.bound (LeftBound39001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9729⟩⟩) (rawTerms := some (Proof.Events152.exact39006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38976 .summary)
      LeftBound38973.bound (LeftBound38973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11979⟩⟩) (rawTerms := some (Proof.Events152.exact38976RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound38973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39001.bound, LeftBound38973.bound]
def bound : CoeffClass := .finite ⟨95450368, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39001.bound, LeftBound38973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39001.actual selector witness, LeftBound38973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39011

namespace LeftBound39015
def owner : Owner := ⟨.program ⟨214⟩, ⟨25230⟩⟩
def transferEvent : Nat := 39015
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39013 .coefficient) (.predecessor 1 39014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39013 .coefficient)
      LeftBound39009.bound (LeftBound39009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39014 .coefficient)
      LeftAuthority38947.bound (LeftAuthority38947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact38948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38947.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39009.bound LeftAuthority38947.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39009.bound, LeftAuthority38947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39009.actual selector witness) * (LeftAuthority38947.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39015

namespace LeftBound39016
def owner : Owner := ⟨.program ⟨214⟩, ⟨25230⟩⟩
def transferEvent : Nat := 39016
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25229⟩⟩]⟩ [⟨.result 38948 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 38948 .coefficient)
      LeftAuthority38947.bound (LeftAuthority38947.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25229⟩⟩) (rawTerms := some (Proof.Events152.exact38948RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority38947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority38947.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority38947.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority38947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority38947.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39016

namespace LeftBound39017
def owner : Owner := ⟨.program ⟨214⟩, ⟨25230⟩⟩
def transferEvent : Nat := 39017
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 39012 .summary) (.transfer 39016) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39012 .summary)
      LeftBound39011.bound (LeftBound39011.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11980⟩⟩) (rawTerms := some (Proof.Events152.exact39012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound39011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39016)
      LeftBound39016.bound (LeftBound39016.actual selector witness) := by
  exact .transfer (LeftBound39016.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound39011.bound LeftBound39016.bound
def bound : CoeffClass := .finite ⟨350304377765888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39011.bound, LeftBound39016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound39011.actual selector witness) * (LeftBound39016.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39017

namespace LeftBound39028
def owner : Owner := ⟨.program ⟨214⟩, ⟨19826⟩⟩
def transferEvent : Nat := 39028
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 39026 .coefficient) (.value (.predecessor 1 39027 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39026 .coefficient)
      LeftAuthority39024.bound (LeftAuthority39024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39024.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39027 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority39024.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39024.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39024.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound39028

namespace LeftBound39032
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def transferEvent : Nat := 39032
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 39030 .coefficient) (.predecessor 1 39031 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39030 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39031 .coefficient)
      LeftBound39028.bound (LeftBound39028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39028.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound39028.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound39028.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound39028.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39032

namespace LeftBound39033
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def transferEvent : Nat := 39033
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19824⟩⟩]⟩ [⟨.result 39025 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 39025 .coefficient)
      LeftAuthority39024.bound (LeftAuthority39024.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19824⟩⟩) (rawTerms := some (Proof.Events152.exact39025RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39024.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39024.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority39024.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority39024.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound39033

namespace LeftBound39034
def owner : Owner := ⟨.program ⟨214⟩, ⟨19827⟩⟩
def transferEvent : Nat := 39034
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 39033) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 39033)
      LeftBound39033.bound (LeftBound39033.actual selector witness) := by
  exact .transfer (LeftBound39033.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound39033.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound39033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound39033.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39034

namespace LeftBound39113
def owner : Owner := ⟨.program ⟨214⟩, ⟨11974⟩⟩
def transferEvent : Nat := 39113
def frameStart : Nat := 39084
def rule : BoundRule := .product (.predecessor 0 39111 .coefficient) (.predecessor 1 39112 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39111 .coefficient)
      LeftAuthority39109.bound (LeftAuthority39109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39112 .coefficient)
      LeftAuthority39106.bound (LeftAuthority39106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority39109.bound LeftAuthority39106.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39109.bound, LeftAuthority39106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority39109.actual selector witness) * (LeftAuthority39106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39113

namespace LeftBound39117
def owner : Owner := ⟨.program ⟨214⟩, ⟨11975⟩⟩
def transferEvent : Nat := 39117
def frameStart : Nat := 39084
def rule : BoundRule := .identity (.predecessor 0 39116 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39116 .coefficient)
      LeftBound39113.bound (LeftBound39113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39113.derived selector witness)

def rawBound : CoeffClass := LeftBound39113.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39113.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39117

namespace LeftBound39134
def owner : Owner := ⟨.program ⟨214⟩, ⟨12061⟩⟩
def transferEvent : Nat := 39134
def frameStart : Nat := 39084
def rule : BoundRule := .sum [.predecessor 0 39132 .coefficient, .predecessor 1 39133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39132 .coefficient)
      LeftBound39117.bound (LeftBound39117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39133 .coefficient)
      LeftAuthority39130.bound (LeftAuthority39130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority39130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound39117.bound, LeftAuthority39130.bound]
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39117.bound, LeftAuthority39130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound39117.actual selector witness, LeftAuthority39130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound39134

namespace LeftBound39137
def owner : Owner := ⟨.program ⟨214⟩, ⟨12062⟩⟩
def transferEvent : Nat := 39137
def frameStart : Nat := 39084
def rule : BoundRule := .identity (.predecessor 0 39136 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39136 .coefficient)
      LeftBound39134.bound (LeftBound39134.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound39134.derived selector witness)

def rawBound : CoeffClass := LeftBound39134.bound
def bound : CoeffClass := .finite ⟨1296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound39134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound39134.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound39137

namespace LeftBound39143
def owner : Owner := ⟨.program ⟨214⟩, ⟨12063⟩⟩
def transferEvent : Nat := 39143
def frameStart : Nat := 39084
def rule : BoundRule := .product (.predecessor 0 39141 .coefficient) (.predecessor 1 39142 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 39141 .coefficient)
      LeftAuthority39139.bound (LeftAuthority39139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority39139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority39139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 39142 .coefficient)
      LeftBound39137.bound (LeftBound39137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events152.exact39138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound39137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound39137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority39139.bound LeftBound39137.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority39139.bound, LeftBound39137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority39139.actual selector witness) * (LeftBound39137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound39143

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
