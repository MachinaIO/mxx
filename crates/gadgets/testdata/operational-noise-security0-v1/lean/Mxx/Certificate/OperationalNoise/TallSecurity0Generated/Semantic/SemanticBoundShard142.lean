import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard140
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard141

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22179
def owner : Owner := ⟨.program ⟨214⟩, ⟨25698⟩⟩
def transferEvent : Nat := 22179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 22173 .summary, .result 21987 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22173 .summary)
      LeftBound21999.bound (LeftBound21999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20191⟩⟩) (rawTerms := some (Proof.Events086.exact22173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21987 .summary)
      LeftBound21982.bound (LeftBound21982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25697⟩⟩) (rawTerms := some (Proof.Events085.exact21987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21999.bound, LeftBound21982.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21999.bound, LeftBound21982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound21999.actual selector witness, LeftBound21982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22179

namespace LeftBound22183
def owner : Owner := ⟨.program ⟨214⟩, ⟨29860⟩⟩
def transferEvent : Nat := 22183
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22181 .coefficient) (.predecessor 1 22182 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22181 .coefficient)
      LeftBound22176.bound (LeftBound22176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22182 .coefficient)
      LeftAuthority21902.bound (LeftAuthority21902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21902.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22176.bound LeftAuthority21902.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22176.bound, LeftAuthority21902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22176.actual selector witness) * (LeftAuthority21902.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22183

namespace LeftBound22184
def owner : Owner := ⟨.program ⟨214⟩, ⟨29860⟩⟩
def transferEvent : Nat := 22184
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29858⟩⟩]⟩ [⟨.result 21903 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21903 .coefficient)
      LeftAuthority21902.bound (LeftAuthority21902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29858⟩⟩) (rawTerms := some (Proof.Events085.exact21903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21902.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority21902.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22184

namespace LeftBound22185
def owner : Owner := ⟨.program ⟨214⟩, ⟨29860⟩⟩
def transferEvent : Nat := 22185
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22180 .summary) (.transfer 22184) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22180 .summary)
      LeftBound22179.bound (LeftBound22179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25698⟩⟩) (rawTerms := some (Proof.Events086.exact22180RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22184)
      LeftBound22184.bound (LeftBound22184.actual selector witness) := by
  exact .transfer (LeftBound22184.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22179.bound LeftBound22184.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22179.bound, LeftBound22184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22179.actual selector witness) * (LeftBound22184.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22185

namespace LeftBound22196
def owner : Owner := ⟨.program ⟨214⟩, ⟨22710⟩⟩
def transferEvent : Nat := 22196
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 22194 .coefficient) (.value (.predecessor 1 22195 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22194 .coefficient)
      LeftAuthority22192.bound (LeftAuthority22192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22195 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22192.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22192.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22192.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22196

namespace LeftBound22200
def owner : Owner := ⟨.program ⟨214⟩, ⟨22711⟩⟩
def transferEvent : Nat := 22200
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22198 .coefficient) (.predecessor 1 22199 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22198 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22199 .coefficient)
      LeftBound22196.bound (LeftBound22196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound22196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound22196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound22196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22200

namespace LeftBound22201
def owner : Owner := ⟨.program ⟨214⟩, ⟨22711⟩⟩
def transferEvent : Nat := 22201
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22708⟩⟩]⟩ [⟨.result 22193 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22193 .coefficient)
      LeftAuthority22192.bound (LeftAuthority22192.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22708⟩⟩) (rawTerms := some (Proof.Events086.exact22193RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22192.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22192.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22192.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22201

namespace LeftBound22202
def owner : Owner := ⟨.program ⟨214⟩, ⟨22711⟩⟩
def transferEvent : Nat := 22202
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 22201) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22201)
      LeftBound22201.bound (LeftBound22201.actual selector witness) := by
  exact .transfer (LeftBound22201.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound22201.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound22201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound22201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22202

namespace LeftBound22297
def owner : Owner := ⟨.program ⟨214⟩, ⟨16884⟩⟩
def transferEvent : Nat := 22297
def frameStart : Nat := 22258
def rule : BoundRule := .identity (.predecessor 0 22296 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22296 .coefficient)
      LeftAuthority22294.bound (LeftAuthority22294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22294.derived selector witness)

def rawBound : CoeffClass := LeftAuthority22294.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority22294.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22297

namespace LeftBound22314
def owner : Owner := ⟨.program ⟨214⟩, ⟨16979⟩⟩
def transferEvent : Nat := 22314
def frameStart : Nat := 22258
def rule : BoundRule := .sum [.predecessor 0 22312 .coefficient, .predecessor 1 22313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22312 .coefficient)
      LeftBound22297.bound (LeftBound22297.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22313 .coefficient)
      LeftAuthority22310.bound (LeftAuthority22310.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority22310.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound22297.bound, LeftAuthority22310.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22297.bound, LeftAuthority22310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound22297.actual selector witness, LeftAuthority22310.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22314

namespace LeftBound22317
def owner : Owner := ⟨.program ⟨214⟩, ⟨16980⟩⟩
def transferEvent : Nat := 22317
def frameStart : Nat := 22258
def rule : BoundRule := .identity (.predecessor 0 22316 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22316 .coefficient)
      LeftBound22314.bound (LeftBound22314.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound22314.derived selector witness)

def rawBound : CoeffClass := LeftBound22314.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound22314.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22317

namespace LeftBound22323
def owner : Owner := ⟨.program ⟨214⟩, ⟨16981⟩⟩
def transferEvent : Nat := 22323
def frameStart : Nat := 22258
def rule : BoundRule := .product (.predecessor 0 22321 .coefficient) (.predecessor 1 22322 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22321 .coefficient)
      LeftAuthority22319.bound (LeftAuthority22319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22322 .coefficient)
      LeftBound22317.bound (LeftBound22317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22317.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority22319.bound LeftBound22317.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22319.bound, LeftBound22317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority22319.actual selector witness) * (LeftBound22317.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22323

namespace LeftBound22331
def owner : Owner := ⟨.program ⟨214⟩, ⟨16982⟩⟩
def transferEvent : Nat := 22331
def frameStart : Nat := 22258
def rule : BoundRule := .sum [.predecessor 0 22329 .coefficient, .predecessor 1 22330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22329 .coefficient)
      LeftAuthority22327.bound (LeftAuthority22327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22330 .coefficient)
      LeftBound22323.bound (LeftBound22323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22323.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22323.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22327.bound, LeftBound22323.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22327.bound, LeftBound22323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22327.actual selector witness, LeftBound22323.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22331

namespace LeftBound22335
def owner : Owner := ⟨.program ⟨214⟩, ⟨29859⟩⟩
def transferEvent : Nat := 22335
def frameStart : Nat := 22258
def rule : BoundRule := .product (.predecessor 0 22333 .coefficient) (.predecessor 1 22334 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22333 .coefficient)
      LeftBound22331.bound (LeftBound22331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22334 .coefficient)
      LeftAuthority22308.bound (LeftAuthority22308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22308.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22331.bound LeftAuthority22308.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22331.bound, LeftAuthority22308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22331.actual selector witness) * (LeftAuthority22308.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22335

namespace LeftBound22346
def owner : Owner := ⟨.program ⟨214⟩, ⟨17095⟩⟩
def transferEvent : Nat := 22346
def frameStart : Nat := 22258
def rule : BoundRule := .product (.predecessor 0 22344 .coefficient) (.predecessor 1 22345 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22344 .coefficient)
      LeftAuthority22319.bound (LeftAuthority22319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22320RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22319.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22345 .coefficient)
      LeftAuthority22342.bound (LeftAuthority22342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22342.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22342.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority22319.bound LeftAuthority22342.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22319.bound, LeftAuthority22342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority22319.actual selector witness) * (LeftAuthority22342.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22346

namespace LeftBound22354
def owner : Owner := ⟨.program ⟨214⟩, ⟨17096⟩⟩
def transferEvent : Nat := 22354
def frameStart : Nat := 22258
def rule : BoundRule := .sum [.predecessor 0 22352 .coefficient, .predecessor 1 22353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22352 .coefficient)
      LeftAuthority22350.bound (LeftAuthority22350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22350.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22353 .coefficient)
      LeftBound22346.bound (LeftBound22346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events087.exact22348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22350.bound, LeftBound22346.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22350.bound, LeftBound22346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority22350.actual selector witness, LeftBound22346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22354

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
