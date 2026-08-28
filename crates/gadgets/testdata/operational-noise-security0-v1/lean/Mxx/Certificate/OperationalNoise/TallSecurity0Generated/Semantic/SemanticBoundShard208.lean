import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard141
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard142
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard207

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31924
def owner : Owner := ⟨.program ⟨214⟩, ⟨30180⟩⟩
def transferEvent : Nat := 31924
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 31919 .summary) (.transfer 31923) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31919 .summary)
      LeftBound31918.bound (LeftBound31918.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30179⟩⟩) (rawTerms := some (Proof.Events124.exact31919RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31923)
      LeftBound31923.bound (LeftBound31923.actual selector witness) := by
  exact .transfer (LeftBound31923.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound31918.bound LeftBound31923.bound
def bound : CoeffClass := .finite ⟨4743639307122182955475140608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31918.bound, LeftBound31923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound31918.actual selector witness) * (LeftBound31923.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31924

namespace LeftBound31939
def owner : Owner := ⟨.program ⟨214⟩, ⟨29853⟩⟩
def transferEvent : Nat := 31939
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31937 .coefficient) (.predecessor 1 31938 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31937 .coefficient)
      LeftBound22176.bound (LeftBound22176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31938 .coefficient)
      LeftAuthority31935.bound (LeftAuthority31935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22176.bound LeftAuthority31935.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22176.bound, LeftAuthority31935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22176.actual selector witness) * (LeftAuthority31935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31939

namespace LeftBound31940
def owner : Owner := ⟨.program ⟨214⟩, ⟨29853⟩⟩
def transferEvent : Nat := 31940
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29851⟩⟩]⟩ [⟨.result 31936 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31936 .coefficient)
      LeftAuthority31935.bound (LeftAuthority31935.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29851⟩⟩) (rawTerms := some (Proof.Events124.exact31936RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31935.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31935.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31935.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31940

namespace LeftBound31941
def owner : Owner := ⟨.program ⟨214⟩, ⟨29853⟩⟩
def transferEvent : Nat := 31941
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22180 .summary) (.transfer 31940) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22180 .summary)
      LeftBound22179.bound (LeftBound22179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25698⟩⟩) (rawTerms := some (Proof.Events086.exact22180RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31940)
      LeftBound31940.bound (LeftBound31940.actual selector witness) := by
  exact .transfer (LeftBound31940.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22179.bound LeftBound31940.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22179.bound, LeftBound31940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22179.actual selector witness) * (LeftBound31940.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31941

namespace LeftBound31952
def owner : Owner := ⟨.program ⟨214⟩, ⟨22638⟩⟩
def transferEvent : Nat := 31952
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 31950 .coefficient) (.value (.predecessor 1 31951 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31950 .coefficient)
      LeftAuthority31948.bound (LeftAuthority31948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31948.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31951 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority31948.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31948.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31948.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound31952

namespace LeftBound31956
def owner : Owner := ⟨.program ⟨214⟩, ⟨22639⟩⟩
def transferEvent : Nat := 31956
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31954 .coefficient) (.predecessor 1 31955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 31954 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 31955 .coefficient)
      LeftBound31952.bound (LeftBound31952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events124.exact31953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound31952.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound31952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound31952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31956

namespace LeftBound31957
def owner : Owner := ⟨.program ⟨214⟩, ⟨22639⟩⟩
def transferEvent : Nat := 31957
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22636⟩⟩]⟩ [⟨.result 31949 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 31949 .coefficient)
      LeftAuthority31948.bound (LeftAuthority31948.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22636⟩⟩) (rawTerms := some (Proof.Events124.exact31949RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31948.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31948.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31948.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority31948.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31957

namespace LeftBound31958
def owner : Owner := ⟨.program ⟨214⟩, ⟨22639⟩⟩
def transferEvent : Nat := 31958
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 31957) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 31957)
      LeftBound31957.bound (LeftBound31957.actual selector witness) := by
  exact .transfer (LeftBound31957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound31957.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound31957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound31957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31958

namespace LeftBound32053
def owner : Owner := ⟨.program ⟨214⟩, ⟨16884⟩⟩
def transferEvent : Nat := 32053
def frameStart : Nat := 32014
def rule : BoundRule := .identity (.predecessor 0 32052 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32052 .coefficient)
      LeftAuthority32050.bound (LeftAuthority32050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32050.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32050.derived selector witness)

def rawBound : CoeffClass := LeftAuthority32050.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority32050.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32053

namespace LeftBound32070
def owner : Owner := ⟨.program ⟨214⟩, ⟨16979⟩⟩
def transferEvent : Nat := 32070
def frameStart : Nat := 32014
def rule : BoundRule := .sum [.predecessor 0 32068 .coefficient, .predecessor 1 32069 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32068 .coefficient)
      LeftBound32053.bound (LeftBound32053.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32053.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32069 .coefficient)
      LeftAuthority32066.bound (LeftAuthority32066.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority32066.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound32053.bound, LeftAuthority32066.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32053.bound, LeftAuthority32066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound32053.actual selector witness, LeftAuthority32066.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32070

namespace LeftBound32073
def owner : Owner := ⟨.program ⟨214⟩, ⟨16980⟩⟩
def transferEvent : Nat := 32073
def frameStart : Nat := 32014
def rule : BoundRule := .identity (.predecessor 0 32072 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32072 .coefficient)
      LeftBound32070.bound (LeftBound32070.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound32070.derived selector witness)

def rawBound : CoeffClass := LeftBound32070.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound32070.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound32073

namespace LeftBound32079
def owner : Owner := ⟨.program ⟨214⟩, ⟨16981⟩⟩
def transferEvent : Nat := 32079
def frameStart : Nat := 32014
def rule : BoundRule := .product (.predecessor 0 32077 .coefficient) (.predecessor 1 32078 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32077 .coefficient)
      LeftAuthority32075.bound (LeftAuthority32075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32078 .coefficient)
      LeftBound32073.bound (LeftBound32073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32073.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority32075.bound LeftBound32073.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32075.bound, LeftBound32073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority32075.actual selector witness) * (LeftBound32073.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32079

namespace LeftBound32087
def owner : Owner := ⟨.program ⟨214⟩, ⟨16982⟩⟩
def transferEvent : Nat := 32087
def frameStart : Nat := 32014
def rule : BoundRule := .sum [.predecessor 0 32085 .coefficient, .predecessor 1 32086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32085 .coefficient)
      LeftAuthority32083.bound (LeftAuthority32083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32086 .coefficient)
      LeftBound32079.bound (LeftBound32079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32083.bound, LeftBound32079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32083.bound, LeftBound32079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32083.actual selector witness, LeftBound32079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32087

namespace LeftBound32091
def owner : Owner := ⟨.program ⟨214⟩, ⟨29852⟩⟩
def transferEvent : Nat := 32091
def frameStart : Nat := 32014
def rule : BoundRule := .product (.predecessor 0 32089 .coefficient) (.predecessor 1 32090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32089 .coefficient)
      LeftBound32087.bound (LeftBound32087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32090 .coefficient)
      LeftAuthority32064.bound (LeftAuthority32064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32064.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32064.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound32087.bound LeftAuthority32064.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32087.bound, LeftAuthority32064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound32087.actual selector witness) * (LeftAuthority32064.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32091

namespace LeftBound32102
def owner : Owner := ⟨.program ⟨214⟩, ⟨16941⟩⟩
def transferEvent : Nat := 32102
def frameStart : Nat := 32014
def rule : BoundRule := .product (.predecessor 0 32100 .coefficient) (.predecessor 1 32101 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32100 .coefficient)
      LeftAuthority32075.bound (LeftAuthority32075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32101 .coefficient)
      LeftAuthority32098.bound (LeftAuthority32098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32098.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority32075.bound LeftAuthority32098.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32075.bound, LeftAuthority32098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority32075.actual selector witness) * (LeftAuthority32098.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound32102

namespace LeftBound32110
def owner : Owner := ⟨.program ⟨214⟩, ⟨16942⟩⟩
def transferEvent : Nat := 32110
def frameStart : Nat := 32014
def rule : BoundRule := .sum [.predecessor 0 32108 .coefficient, .predecessor 1 32109 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 32108 .coefficient)
      LeftAuthority32106.bound (LeftAuthority32106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority32106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority32106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 32109 .coefficient)
      LeftBound32102.bound (LeftBound32102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority32106.bound, LeftBound32102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority32106.bound, LeftBound32102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority32106.actual selector witness, LeftBound32102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound32110

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
