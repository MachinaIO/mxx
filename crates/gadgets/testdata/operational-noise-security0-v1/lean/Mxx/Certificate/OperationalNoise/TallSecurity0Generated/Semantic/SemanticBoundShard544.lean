import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound80179
def owner : Owner := ⟨.program ⟨214⟩, ⟨25762⟩⟩
def transferEvent : Nat := 80179
def frameStart : Nat := 80067
def rule : BoundRule := .sum [.predecessor 0 80177 .coefficient, .predecessor 1 80178 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80177 .coefficient)
      LeftBound80175.bound (LeftBound80175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80176RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80178 .coefficient)
      LeftBound80156.bound (LeftBound80156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80156.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80175.bound, LeftBound80156.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80175.bound, LeftBound80156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80175.actual selector witness, LeftBound80156.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80179

namespace LeftBound80192
def owner : Owner := ⟨.program ⟨214⟩, ⟨25760⟩⟩
def transferEvent : Nat := 80192
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80190 .coefficient, .predecessor 1 80191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80190 .coefficient)
      LeftBound80015.bound (LeftBound80015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80191 .coefficient)
      LeftBound79987.bound (LeftBound79987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79987.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80015.bound, LeftBound79987.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80015.bound, LeftBound79987.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80015.actual selector witness, LeftBound79987.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80192

namespace LeftBound80195
def owner : Owner := ⟨.program ⟨214⟩, ⟨25760⟩⟩
def transferEvent : Nat := 80195
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80189 .summary, .result 79994 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80189 .summary)
      LeftBound80017.bound (LeftBound80017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20251⟩⟩) (rawTerms := some (Proof.Events313.exact80189RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80017.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79994 .summary)
      LeftBound79989.bound (LeftBound79989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25759⟩⟩) (rawTerms := some (Proof.Events312.exact79994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79989.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80017.bound, LeftBound79989.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80017.bound, LeftBound79989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80017.actual selector witness, LeftBound79989.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80195

namespace LeftBound80199
def owner : Owner := ⟨.program ⟨214⟩, ⟨30118⟩⟩
def transferEvent : Nat := 80199
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80197 .coefficient) (.predecessor 1 80198 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80197 .coefficient)
      LeftBound80192.bound (LeftBound80192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80198 .coefficient)
      LeftAuthority79904.bound (LeftAuthority79904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79904.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80192.bound LeftAuthority79904.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80192.bound, LeftAuthority79904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80192.actual selector witness) * (LeftAuthority79904.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80199

namespace LeftBound80200
def owner : Owner := ⟨.program ⟨214⟩, ⟨30118⟩⟩
def transferEvent : Nat := 80200
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30116⟩⟩]⟩ [⟨.result 79905 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79905 .coefficient)
      LeftAuthority79904.bound (LeftAuthority79904.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30116⟩⟩) (rawTerms := some (Proof.Events312.exact79905RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79904.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79904.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79904.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79904.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80200

namespace LeftBound80201
def owner : Owner := ⟨.program ⟨214⟩, ⟨30118⟩⟩
def transferEvent : Nat := 80201
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80196 .summary) (.transfer 80200) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80196 .summary)
      LeftBound80195.bound (LeftBound80195.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25760⟩⟩) (rawTerms := some (Proof.Events313.exact80196RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80200)
      LeftBound80200.bound (LeftBound80200.actual selector witness) := by
  exact .transfer (LeftBound80200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80195.bound LeftBound80200.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80195.bound, LeftBound80200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80195.actual selector witness) * (LeftBound80200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80201

namespace LeftBound80212
def owner : Owner := ⟨.program ⟨214⟩, ⟨22842⟩⟩
def transferEvent : Nat := 80212
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 80210 .coefficient) (.value (.predecessor 1 80211 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80210 .coefficient)
      LeftAuthority80208.bound (LeftAuthority80208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80211 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority80208.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80208.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80208.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80212

namespace LeftBound80216
def owner : Owner := ⟨.program ⟨214⟩, ⟨22843⟩⟩
def transferEvent : Nat := 80216
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80214 .coefficient) (.predecessor 1 80215 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80214 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80215 .coefficient)
      LeftBound80212.bound (LeftBound80212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80212.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound80212.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound80212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound80212.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80216

namespace LeftBound80217
def owner : Owner := ⟨.program ⟨214⟩, ⟨22843⟩⟩
def transferEvent : Nat := 80217
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22840⟩⟩]⟩ [⟨.result 80209 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80209 .coefficient)
      LeftAuthority80208.bound (LeftAuthority80208.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22840⟩⟩) (rawTerms := some (Proof.Events313.exact80209RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80208.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority80208.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority80208.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound80217

namespace LeftBound80218
def owner : Owner := ⟨.program ⟨214⟩, ⟨22843⟩⟩
def transferEvent : Nat := 80218
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 80217) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 80217)
      LeftBound80217.bound (LeftBound80217.actual selector witness) := by
  exact .transfer (LeftBound80217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound80217.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound80217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound80217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80218

namespace LeftBound80313
def owner : Owner := ⟨.program ⟨214⟩, ⟨17012⟩⟩
def transferEvent : Nat := 80313
def frameStart : Nat := 80274
def rule : BoundRule := .identity (.predecessor 0 80312 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80312 .coefficient)
      LeftAuthority80310.bound (LeftAuthority80310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80310.derived selector witness)

def rawBound : CoeffClass := LeftAuthority80310.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority80310.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80313

namespace LeftBound80330
def owner : Owner := ⟨.program ⟨214⟩, ⟨17051⟩⟩
def transferEvent : Nat := 80330
def frameStart : Nat := 80274
def rule : BoundRule := .sum [.predecessor 0 80328 .coefficient, .predecessor 1 80329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80328 .coefficient)
      LeftBound80313.bound (LeftBound80313.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80329 .coefficient)
      LeftAuthority80326.bound (LeftAuthority80326.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority80326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80313.bound, LeftAuthority80326.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80313.bound, LeftAuthority80326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80313.actual selector witness, LeftAuthority80326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80330

namespace LeftBound80333
def owner : Owner := ⟨.program ⟨214⟩, ⟨17052⟩⟩
def transferEvent : Nat := 80333
def frameStart : Nat := 80274
def rule : BoundRule := .identity (.predecessor 0 80332 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80332 .coefficient)
      LeftBound80330.bound (LeftBound80330.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound80330.derived selector witness)

def rawBound : CoeffClass := LeftBound80330.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound80330.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound80333

namespace LeftBound80339
def owner : Owner := ⟨.program ⟨214⟩, ⟨17053⟩⟩
def transferEvent : Nat := 80339
def frameStart : Nat := 80274
def rule : BoundRule := .product (.predecessor 0 80337 .coefficient) (.predecessor 1 80338 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80337 .coefficient)
      LeftAuthority80335.bound (LeftAuthority80335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80335.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80338 .coefficient)
      LeftBound80333.bound (LeftBound80333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80333.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority80335.bound LeftBound80333.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80335.bound, LeftBound80333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority80335.actual selector witness) * (LeftBound80333.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80339

namespace LeftBound80347
def owner : Owner := ⟨.program ⟨214⟩, ⟨17054⟩⟩
def transferEvent : Nat := 80347
def frameStart : Nat := 80274
def rule : BoundRule := .sum [.predecessor 0 80345 .coefficient, .predecessor 1 80346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80345 .coefficient)
      LeftAuthority80343.bound (LeftAuthority80343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80343.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80346 .coefficient)
      LeftBound80339.bound (LeftBound80339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80339.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80339.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80343.bound, LeftBound80339.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80343.bound, LeftBound80339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority80343.actual selector witness, LeftBound80339.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80347

namespace LeftBound80351
def owner : Owner := ⟨.program ⟨214⟩, ⟨30117⟩⟩
def transferEvent : Nat := 80351
def frameStart : Nat := 80274
def rule : BoundRule := .product (.predecessor 0 80349 .coefficient) (.predecessor 1 80350 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80349 .coefficient)
      LeftBound80347.bound (LeftBound80347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80350 .coefficient)
      LeftAuthority80324.bound (LeftAuthority80324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events313.exact80325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound80347.bound LeftAuthority80324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80347.bound, LeftAuthority80324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound80347.actual selector witness) * (LeftAuthority80324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80351

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
