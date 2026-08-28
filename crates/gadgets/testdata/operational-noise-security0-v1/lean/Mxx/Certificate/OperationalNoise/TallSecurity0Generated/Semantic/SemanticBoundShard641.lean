import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard640

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound94431
def owner : Owner := ⟨.program ⟨214⟩, ⟨13333⟩⟩
def transferEvent : Nat := 94431
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94429 .coefficient, .predecessor 1 94430 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94429 .coefficient)
      LeftBound94421.bound (LeftBound94421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94430 .coefficient)
      LeftBound94393.bound (LeftBound94393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94421.bound, LeftBound94393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94421.bound, LeftBound94393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94421.actual selector witness, LeftBound94393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94431

namespace LeftBound94433
def owner : Owner := ⟨.program ⟨214⟩, ⟨13333⟩⟩
def transferEvent : Nat := 94433
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 94428 .summary, .result 94398 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94428 .summary)
      LeftBound94423.bound (LeftBound94423.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10334⟩⟩) (rawTerms := some (Proof.Events368.exact94428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94423.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94398 .summary)
      LeftBound94395.bound (LeftBound94395.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13332⟩⟩) (rawTerms := some (Proof.Events368.exact94398RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94395.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94423.bound, LeftBound94395.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94423.bound, LeftBound94395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94423.actual selector witness, LeftBound94395.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94433

namespace LeftBound94437
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def transferEvent : Nat := 94437
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94435 .coefficient) (.predecessor 1 94436 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94435 .coefficient)
      LeftBound94431.bound (LeftBound94431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94436 .coefficient)
      LeftAuthority94369.bound (LeftAuthority94369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94431.bound LeftAuthority94369.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94431.bound, LeftAuthority94369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94431.actual selector witness) * (LeftAuthority94369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94437

namespace LeftBound94438
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def transferEvent : Nat := 94438
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25745⟩⟩]⟩ [⟨.result 94370 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94370 .coefficient)
      LeftAuthority94369.bound (LeftAuthority94369.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25745⟩⟩) (rawTerms := some (Proof.Events368.exact94370RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94369.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94369.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94369.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94369.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94438

namespace LeftBound94439
def owner : Owner := ⟨.program ⟨214⟩, ⟨25746⟩⟩
def transferEvent : Nat := 94439
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94434 .summary) (.transfer 94438) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94434 .summary)
      LeftBound94433.bound (LeftBound94433.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13333⟩⟩) (rawTerms := some (Proof.Events368.exact94434RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94438)
      LeftBound94438.bound (LeftBound94438.actual selector witness) := by
  exact .transfer (LeftBound94438.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound94433.bound LeftBound94438.bound
def bound : CoeffClass := .finite ⟨350377660645376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94433.bound, LeftBound94438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound94433.actual selector witness) * (LeftBound94438.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94439

namespace LeftBound94450
def owner : Owner := ⟨.program ⟨214⟩, ⟨20239⟩⟩
def transferEvent : Nat := 94450
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 94448 .coefficient) (.value (.predecessor 1 94449 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94448 .coefficient)
      LeftAuthority94446.bound (LeftAuthority94446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94449 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority94446.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94446.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94446.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound94450

namespace LeftBound94454
def owner : Owner := ⟨.program ⟨214⟩, ⟨5508⟩⟩
def transferEvent : Nat := 94454
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94452 .coefficient) (.predecessor 1 94453 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94452 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94453 .coefficient)
      LeftAuthority6549.bound (LeftAuthority6549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 16) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94454

namespace LeftBound94459
def owner : Owner := ⟨.program ⟨214⟩, ⟨5509⟩⟩
def transferEvent : Nat := 94459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 94457 .coefficient, .predecessor 1 94458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94457 .coefficient)
      LeftBound94454.bound (LeftBound94454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94454.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94458 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94454.bound, LeftAuthority6547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94454.bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94454.actual selector witness, LeftAuthority6547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94459

namespace LeftBound94460
def owner : Owner := ⟨.program ⟨214⟩, ⟨5509⟩⟩
def transferEvent : Nat := 94460
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩ [⟨.result 6548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6548 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22⟩⟩) (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6547.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94460

namespace LeftBound94465
def owner : Owner := ⟨.program ⟨214⟩, ⟨20240⟩⟩
def transferEvent : Nat := 94465
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 94463 .coefficient) (.predecessor 1 94464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94463 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94464 .coefficient)
      LeftBound94450.bound (LeftBound94450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94450.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound94450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound94450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound94450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94465

namespace LeftBound94466
def owner : Owner := ⟨.program ⟨214⟩, ⟨20240⟩⟩
def transferEvent : Nat := 94466
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20237⟩⟩]⟩ [⟨.result 94447 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94447 .coefficient)
      LeftAuthority94446.bound (LeftAuthority94446.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20237⟩⟩) (rawTerms := some (Proof.Events368.exact94447RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94446.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority94446.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority94446.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound94466

namespace LeftBound94467
def owner : Owner := ⟨.program ⟨214⟩, ⟨20240⟩⟩
def transferEvent : Nat := 94467
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 94466) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 94466)
      LeftBound94466.bound (LeftBound94466.actual selector witness) := by
  exact .transfer (LeftBound94466.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound94466.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound94466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound94466.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94467

namespace LeftBound94522
def owner : Owner := ⟨.program ⟨214⟩, ⟨13327⟩⟩
def transferEvent : Nat := 94522
def frameStart : Nat := 94505
def rule : BoundRule := .product (.predecessor 0 94520 .coefficient) (.predecessor 1 94521 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94520 .coefficient)
      LeftAuthority94518.bound (LeftAuthority94518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94518.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94518.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94521 .coefficient)
      LeftAuthority94515.bound (LeftAuthority94515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority94515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority94515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority94518.bound LeftAuthority94515.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority94518.bound, LeftAuthority94515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority94518.actual selector witness) * (LeftAuthority94515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound94522

namespace LeftBound94526
def owner : Owner := ⟨.program ⟨214⟩, ⟨13328⟩⟩
def transferEvent : Nat := 94526
def frameStart : Nat := 94505
def rule : BoundRule := .identity (.predecessor 0 94525 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94525 .coefficient)
      LeftBound94522.bound (LeftBound94522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events369.exact94524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94522.derived selector witness)

def rawBound : CoeffClass := LeftBound94522.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94522.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound94522.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94526

namespace LeftBound94543
def owner : Owner := ⟨.program ⟨214⟩, ⟨13438⟩⟩
def transferEvent : Nat := 94543
def frameStart : Nat := 94505
def rule : BoundRule := .sum [.predecessor 0 94541 .coefficient, .predecessor 1 94542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94541 .coefficient)
      LeftBound94526.bound (LeftBound94526.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound94526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 94542 .coefficient)
      LeftAuthority94539.bound (LeftAuthority94539.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority94539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound94526.bound, LeftAuthority94539.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94526.bound, LeftAuthority94539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound94526.actual selector witness, LeftAuthority94539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound94543

namespace LeftBound94546
def owner : Owner := ⟨.program ⟨214⟩, ⟨13439⟩⟩
def transferEvent : Nat := 94546
def frameStart : Nat := 94505
def rule : BoundRule := .identity (.predecessor 0 94545 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 94545 .coefficient)
      LeftBound94543.bound (LeftBound94543.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound94543.derived selector witness)

def rawBound : CoeffClass := LeftBound94543.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound94543.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound94546

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
