import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard358

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53314
def owner : Owner := ⟨.program ⟨214⟩, ⟨12473⟩⟩
def transferEvent : Nat := 53314
def frameStart : Nat := 53227
def rule : BoundRule := .sum [.predecessor 0 53312 .coefficient, .predecessor 1 53313 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53312 .coefficient)
      LeftBound53309.bound (LeftBound53309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53313 .coefficient)
      LeftBound53286.bound (LeftBound53286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53288RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53309.bound, LeftBound53286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53309.bound, LeftBound53286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53309.actual selector witness, LeftBound53286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53314

namespace LeftBound53318
def owner : Owner := ⟨.program ⟨214⟩, ⟨25381⟩⟩
def transferEvent : Nat := 53318
def frameStart : Nat := 53227
def rule : BoundRule := .product (.predecessor 0 53316 .coefficient) (.predecessor 1 53317 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53316 .coefficient)
      LeftBound53314.bound (LeftBound53314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53317 .coefficient)
      LeftAuthority53271.bound (LeftAuthority53271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53314.bound LeftAuthority53271.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53314.bound, LeftAuthority53271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53314.actual selector witness) * (LeftAuthority53271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53318

namespace LeftBound53329
def owner : Owner := ⟨.program ⟨214⟩, ⟨16471⟩⟩
def transferEvent : Nat := 53329
def frameStart : Nat := 53227
def rule : BoundRule := .product (.predecessor 0 53327 .coefficient) (.predecessor 1 53328 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53327 .coefficient)
      LeftAuthority53282.bound (LeftAuthority53282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53328 .coefficient)
      LeftAuthority53325.bound (LeftAuthority53325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53325.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53282.bound LeftAuthority53325.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53282.bound, LeftAuthority53325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53282.actual selector witness) * (LeftAuthority53325.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53329

namespace LeftBound53337
def owner : Owner := ⟨.program ⟨214⟩, ⟨16472⟩⟩
def transferEvent : Nat := 53337
def frameStart : Nat := 53227
def rule : BoundRule := .sum [.predecessor 0 53335 .coefficient, .predecessor 1 53336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53335 .coefficient)
      LeftAuthority53333.bound (LeftAuthority53333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53333.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53336 .coefficient)
      LeftBound53329.bound (LeftBound53329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53333.bound, LeftBound53329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53333.bound, LeftBound53329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53333.actual selector witness, LeftBound53329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53337

namespace LeftBound53341
def owner : Owner := ⟨.program ⟨214⟩, ⟨25382⟩⟩
def transferEvent : Nat := 53341
def frameStart : Nat := 53227
def rule : BoundRule := .sum [.predecessor 0 53339 .coefficient, .predecessor 1 53340 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53339 .coefficient)
      LeftBound53337.bound (LeftBound53337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53340 .coefficient)
      LeftBound53318.bound (LeftBound53318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53337.bound, LeftBound53318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53337.bound, LeftBound53318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53337.actual selector witness, LeftBound53318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53341

namespace LeftBound53354
def owner : Owner := ⟨.program ⟨214⟩, ⟨25380⟩⟩
def transferEvent : Nat := 53354
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53352 .coefficient, .predecessor 1 53353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53352 .coefficient)
      LeftBound53175.bound (LeftBound53175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53353 .coefficient)
      LeftBound53158.bound (LeftBound53158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53175.bound, LeftBound53158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53175.bound, LeftBound53158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53175.actual selector witness, LeftBound53158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53354

namespace LeftBound53357
def owner : Owner := ⟨.program ⟨214⟩, ⟨25380⟩⟩
def transferEvent : Nat := 53357
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53351 .summary, .result 53165 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53351 .summary)
      LeftBound53177.bound (LeftBound53177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19895⟩⟩) (rawTerms := some (Proof.Events208.exact53351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53165 .summary)
      LeftBound53160.bound (LeftBound53160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25379⟩⟩) (rawTerms := some (Proof.Events207.exact53165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53160.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53177.bound, LeftBound53160.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53177.bound, LeftBound53160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53177.actual selector witness, LeftBound53160.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53357

namespace LeftBound53361
def owner : Owner := ⟨.program ⟨214⟩, ⟨28966⟩⟩
def transferEvent : Nat := 53361
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53359 .coefficient) (.predecessor 1 53360 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53359 .coefficient)
      LeftBound53354.bound (LeftBound53354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53360 .coefficient)
      LeftAuthority53080.bound (LeftAuthority53080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events207.exact53081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53354.bound LeftAuthority53080.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53354.bound, LeftAuthority53080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53354.actual selector witness) * (LeftAuthority53080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53361

namespace LeftBound53362
def owner : Owner := ⟨.program ⟨214⟩, ⟨28966⟩⟩
def transferEvent : Nat := 53362
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28964⟩⟩]⟩ [⟨.result 53081 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53081 .coefficient)
      LeftAuthority53080.bound (LeftAuthority53080.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28964⟩⟩) (rawTerms := some (Proof.Events207.exact53081RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53080.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53080.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53080.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53080.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53362

namespace LeftBound53363
def owner : Owner := ⟨.program ⟨214⟩, ⟨28966⟩⟩
def transferEvent : Nat := 53363
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 53358 .summary) (.transfer 53362) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53358 .summary)
      LeftBound53357.bound (LeftBound53357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25380⟩⟩) (rawTerms := some (Proof.Events208.exact53358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53362)
      LeftBound53362.bound (LeftBound53362.actual selector witness) := by
  exact .transfer (LeftBound53362.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53357.bound LeftBound53362.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53357.bound, LeftBound53362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53357.actual selector witness) * (LeftBound53362.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53363

namespace LeftBound53374
def owner : Owner := ⟨.program ⟨214⟩, ⟨22126⟩⟩
def transferEvent : Nat := 53374
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 53372 .coefficient) (.value (.predecessor 1 53373 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53372 .coefficient)
      LeftAuthority53370.bound (LeftAuthority53370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53373 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority53370.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53370.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53370.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53374

namespace LeftBound53378
def owner : Owner := ⟨.program ⟨214⟩, ⟨22127⟩⟩
def transferEvent : Nat := 53378
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53376 .coefficient) (.predecessor 1 53377 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53376 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53377 .coefficient)
      LeftBound53374.bound (LeftBound53374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53374.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound53374.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound53374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound53374.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53378

namespace LeftBound53379
def owner : Owner := ⟨.program ⟨214⟩, ⟨22127⟩⟩
def transferEvent : Nat := 53379
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22124⟩⟩]⟩ [⟨.result 53371 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53371 .coefficient)
      LeftAuthority53370.bound (LeftAuthority53370.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22124⟩⟩) (rawTerms := some (Proof.Events208.exact53371RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53370.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53370.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority53370.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53379

namespace LeftBound53380
def owner : Owner := ⟨.program ⟨214⟩, ⟨22127⟩⟩
def transferEvent : Nat := 53380
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 53379) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 53379)
      LeftBound53379.bound (LeftBound53379.actual selector witness) := by
  exact .transfer (LeftBound53379.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound53379.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound53379.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound53379.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53380

namespace LeftBound53475
def owner : Owner := ⟨.program ⟨214⟩, ⟨16470⟩⟩
def transferEvent : Nat := 53475
def frameStart : Nat := 53436
def rule : BoundRule := .identity (.predecessor 0 53474 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53474 .coefficient)
      LeftAuthority53472.bound (LeftAuthority53472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53472.derived selector witness)

def rawBound : CoeffClass := LeftAuthority53472.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53472.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority53472.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53475

namespace LeftBound53492
def owner : Owner := ⟨.program ⟨214⟩, ⟨16509⟩⟩
def transferEvent : Nat := 53492
def frameStart : Nat := 53436
def rule : BoundRule := .sum [.predecessor 0 53490 .coefficient, .predecessor 1 53491 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53490 .coefficient)
      LeftBound53475.bound (LeftBound53475.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53491 .coefficient)
      LeftAuthority53488.bound (LeftAuthority53488.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority53488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53475.bound, LeftAuthority53488.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53475.bound, LeftAuthority53488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53475.actual selector witness, LeftAuthority53488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53492

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
