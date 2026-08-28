import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard558

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound82137
def owner : Owner := ⟨.program ⟨214⟩, ⟨22267⟩⟩
def transferEvent : Nat := 82137
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22264⟩⟩]⟩ [⟨.result 82129 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82129 .coefficient)
      LeftAuthority82128.bound (LeftAuthority82128.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22264⟩⟩) (rawTerms := some (Proof.Events320.exact82129RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82128.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority82128.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority82128.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound82137

namespace LeftBound82138
def owner : Owner := ⟨.program ⟨214⟩, ⟨22267⟩⟩
def transferEvent : Nat := 82138
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 82137) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 82137)
      LeftBound82137.bound (LeftBound82137.actual selector witness) := by
  exact .transfer (LeftBound82137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound82137.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound82137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound82137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82138

namespace LeftBound82233
def owner : Owner := ⟨.program ⟨214⟩, ⟨16550⟩⟩
def transferEvent : Nat := 82233
def frameStart : Nat := 82194
def rule : BoundRule := .identity (.predecessor 0 82232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82232 .coefficient)
      LeftAuthority82230.bound (LeftAuthority82230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82230.derived selector witness)

def rawBound : CoeffClass := LeftAuthority82230.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority82230.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82233

namespace LeftBound82250
def owner : Owner := ⟨.program ⟨214⟩, ⟨16589⟩⟩
def transferEvent : Nat := 82250
def frameStart : Nat := 82194
def rule : BoundRule := .sum [.predecessor 0 82248 .coefficient, .predecessor 1 82249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82248 .coefficient)
      LeftBound82233.bound (LeftBound82233.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82249 .coefficient)
      LeftAuthority82246.bound (LeftAuthority82246.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority82246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82233.bound, LeftAuthority82246.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82233.bound, LeftAuthority82246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82233.actual selector witness, LeftAuthority82246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82250

namespace LeftBound82253
def owner : Owner := ⟨.program ⟨214⟩, ⟨16590⟩⟩
def transferEvent : Nat := 82253
def frameStart : Nat := 82194
def rule : BoundRule := .identity (.predecessor 0 82252 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82252 .coefficient)
      LeftBound82250.bound (LeftBound82250.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound82250.derived selector witness)

def rawBound : CoeffClass := LeftBound82250.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82250.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound82250.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound82253

namespace LeftBound82259
def owner : Owner := ⟨.program ⟨214⟩, ⟨16591⟩⟩
def transferEvent : Nat := 82259
def frameStart : Nat := 82194
def rule : BoundRule := .product (.predecessor 0 82257 .coefficient) (.predecessor 1 82258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82257 .coefficient)
      LeftAuthority82255.bound (LeftAuthority82255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82258 .coefficient)
      LeftBound82253.bound (LeftBound82253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority82255.bound LeftBound82253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82255.bound, LeftBound82253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority82255.actual selector witness) * (LeftBound82253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82259

namespace LeftBound82267
def owner : Owner := ⟨.program ⟨214⟩, ⟨16592⟩⟩
def transferEvent : Nat := 82267
def frameStart : Nat := 82194
def rule : BoundRule := .sum [.predecessor 0 82265 .coefficient, .predecessor 1 82266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82265 .coefficient)
      LeftAuthority82263.bound (LeftAuthority82263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82263.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82266 .coefficient)
      LeftBound82259.bound (LeftBound82259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82263.bound, LeftBound82259.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82263.bound, LeftBound82259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82263.actual selector witness, LeftBound82259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82267

namespace LeftBound82271
def owner : Owner := ⟨.program ⟨214⟩, ⟨29169⟩⟩
def transferEvent : Nat := 82271
def frameStart : Nat := 82194
def rule : BoundRule := .product (.predecessor 0 82269 .coefficient) (.predecessor 1 82270 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82269 .coefficient)
      LeftBound82267.bound (LeftBound82267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82270 .coefficient)
      LeftAuthority82244.bound (LeftAuthority82244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82244.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82267.bound LeftAuthority82244.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82267.bound, LeftAuthority82244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82267.actual selector witness) * (LeftAuthority82244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82271

namespace LeftBound82282
def owner : Owner := ⟨.program ⟨214⟩, ⟨18206⟩⟩
def transferEvent : Nat := 82282
def frameStart : Nat := 82194
def rule : BoundRule := .product (.predecessor 0 82280 .coefficient) (.predecessor 1 82281 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82280 .coefficient)
      LeftAuthority82255.bound (LeftAuthority82255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82281 .coefficient)
      LeftAuthority82278.bound (LeftAuthority82278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82278.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority82255.bound LeftAuthority82278.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82255.bound, LeftAuthority82278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority82255.actual selector witness) * (LeftAuthority82278.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82282

namespace LeftBound82290
def owner : Owner := ⟨.program ⟨214⟩, ⟨18207⟩⟩
def transferEvent : Nat := 82290
def frameStart : Nat := 82194
def rule : BoundRule := .sum [.predecessor 0 82288 .coefficient, .predecessor 1 82289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82288 .coefficient)
      LeftAuthority82286.bound (LeftAuthority82286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority82286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority82286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82289 .coefficient)
      LeftBound82282.bound (LeftBound82282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority82286.bound, LeftBound82282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority82286.bound, LeftBound82282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority82286.actual selector witness, LeftBound82282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82290

namespace LeftBound82294
def owner : Owner := ⟨.program ⟨214⟩, ⟨29173⟩⟩
def transferEvent : Nat := 82294
def frameStart : Nat := 82194
def rule : BoundRule := .sum [.predecessor 0 82292 .coefficient, .predecessor 1 82293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82292 .coefficient)
      LeftBound82290.bound (LeftBound82290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82293 .coefficient)
      LeftBound82271.bound (LeftBound82271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82290.bound, LeftBound82271.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82290.bound, LeftBound82271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82290.actual selector witness, LeftBound82271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82294

namespace LeftBound82307
def owner : Owner := ⟨.program ⟨214⟩, ⟨29171⟩⟩
def transferEvent : Nat := 82307
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82305 .coefficient, .predecessor 1 82306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82305 .coefficient)
      LeftBound82136.bound (LeftBound82136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82306 .coefficient)
      LeftBound82119.bound (LeftBound82119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events320.exact82126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82136.bound, LeftBound82119.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82136.bound, LeftBound82119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82136.actual selector witness, LeftBound82119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82307

namespace LeftBound82310
def owner : Owner := ⟨.program ⟨214⟩, ⟨29171⟩⟩
def transferEvent : Nat := 82310
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 82304 .summary, .result 82126 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82304 .summary)
      LeftBound82138.bound (LeftBound82138.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22267⟩⟩) (rawTerms := some (Proof.Events321.exact82304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82126 .summary)
      LeftBound82121.bound (LeftBound82121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29170⟩⟩) (rawTerms := some (Proof.Events320.exact82126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82138.bound, LeftBound82121.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82138.bound, LeftBound82121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82138.actual selector witness, LeftBound82121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82310

namespace LeftBound82334
def owner : Owner := ⟨.program ⟨214⟩, ⟨12373⟩⟩
def transferEvent : Nat := 82334
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 82332 .coefficient) (.predecessor 1 82333 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82332 .coefficient)
      LeftAuthority3942.bound (LeftAuthority3942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82333 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3942.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3942.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3942.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound82334

namespace LeftBound82339
def owner : Owner := ⟨.program ⟨214⟩, ⟨7241⟩⟩
def transferEvent : Nat := 82339
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 82337 .coefficient) (.predecessor 1 82338 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82337 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82338 .coefficient)
      LeftBound8976.bound (LeftBound8976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound8976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound8976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound8976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound82339

namespace LeftBound82344
def owner : Owner := ⟨.program ⟨214⟩, ⟨12374⟩⟩
def transferEvent : Nat := 82344
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 82342 .coefficient, .predecessor 1 82343 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 82342 .coefficient)
      LeftBound82339.bound (LeftBound82339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 82343 .coefficient)
      LeftBound82334.bound (LeftBound82334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events321.exact82336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82334.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound82339.bound, LeftBound82334.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82339.bound, LeftBound82334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound82339.actual selector witness, LeftBound82334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound82344

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
