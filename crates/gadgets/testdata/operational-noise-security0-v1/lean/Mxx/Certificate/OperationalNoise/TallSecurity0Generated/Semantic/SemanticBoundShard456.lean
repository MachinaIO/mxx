import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard455

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound67303
def owner : Owner := ⟨.program ⟨214⟩, ⟨25446⟩⟩
def transferEvent : Nat := 67303
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67298 .summary) (.transfer 67302) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67298 .summary)
      LeftBound67297.bound (LeftBound67297.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12565⟩⟩) (rawTerms := some (Proof.Events262.exact67298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67302)
      LeftBound67302.bound (LeftBound67302.actual selector witness) := by
  exact .transfer (LeftBound67302.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67297.bound LeftBound67302.bound
def bound : CoeffClass := .finite ⟨350322698485760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67297.bound, LeftBound67302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67297.actual selector witness) * (LeftBound67302.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67303

namespace LeftBound67314
def owner : Owner := ⟨.program ⟨214⟩, ⟨19958⟩⟩
def transferEvent : Nat := 67314
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 67312 .coefficient) (.value (.predecessor 1 67313 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67312 .coefficient)
      LeftAuthority67310.bound (LeftAuthority67310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67313 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67310.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67310.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67310.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67314

namespace LeftBound67318
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def transferEvent : Nat := 67318
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 67316 .coefficient) (.predecessor 1 67317 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67316 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67317 .coefficient)
      LeftBound67314.bound (LeftBound67314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events262.exact67315RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67314.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound67314.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound67314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound67314.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67318

namespace LeftBound67319
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def transferEvent : Nat := 67319
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19956⟩⟩]⟩ [⟨.result 67311 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67311 .coefficient)
      LeftAuthority67310.bound (LeftAuthority67310.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19956⟩⟩) (rawTerms := some (Proof.Events262.exact67311RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67310.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority67310.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67310.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound67319

namespace LeftBound67320
def owner : Owner := ⟨.program ⟨214⟩, ⟨19959⟩⟩
def transferEvent : Nat := 67320
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 67319) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 67319)
      LeftBound67319.bound (LeftBound67319.actual selector witness) := by
  exact .transfer (LeftBound67319.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound67319.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound67319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound67319.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67320

namespace LeftBound67399
def owner : Owner := ⟨.program ⟨214⟩, ⟨12559⟩⟩
def transferEvent : Nat := 67399
def frameStart : Nat := 67370
def rule : BoundRule := .product (.predecessor 0 67397 .coefficient) (.predecessor 1 67398 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67397 .coefficient)
      LeftAuthority67395.bound (LeftAuthority67395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67398 .coefficient)
      LeftAuthority67392.bound (LeftAuthority67392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67392.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67392.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67395.bound LeftAuthority67392.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67395.bound, LeftAuthority67392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67395.actual selector witness) * (LeftAuthority67392.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67399

namespace LeftBound67403
def owner : Owner := ⟨.program ⟨214⟩, ⟨12560⟩⟩
def transferEvent : Nat := 67403
def frameStart : Nat := 67370
def rule : BoundRule := .identity (.predecessor 0 67402 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67402 .coefficient)
      LeftBound67399.bound (LeftBound67399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67399.derived selector witness)

def rawBound : CoeffClass := LeftBound67399.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67399.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67403

namespace LeftBound67420
def owner : Owner := ⟨.program ⟨214⟩, ⟨12658⟩⟩
def transferEvent : Nat := 67420
def frameStart : Nat := 67370
def rule : BoundRule := .sum [.predecessor 0 67418 .coefficient, .predecessor 1 67419 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67418 .coefficient)
      LeftBound67403.bound (LeftBound67403.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67403.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67419 .coefficient)
      LeftAuthority67416.bound (LeftAuthority67416.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67403.bound, LeftAuthority67416.bound]
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67403.bound, LeftAuthority67416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67403.actual selector witness, LeftAuthority67416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67420

namespace LeftBound67423
def owner : Owner := ⟨.program ⟨214⟩, ⟨12659⟩⟩
def transferEvent : Nat := 67423
def frameStart : Nat := 67370
def rule : BoundRule := .identity (.predecessor 0 67422 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67422 .coefficient)
      LeftBound67420.bound (LeftBound67420.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound67420.derived selector witness)

def rawBound : CoeffClass := LeftBound67420.bound
def bound : CoeffClass := .finite ⟨1764, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound67420.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67423

namespace LeftBound67429
def owner : Owner := ⟨.program ⟨214⟩, ⟨12660⟩⟩
def transferEvent : Nat := 67429
def frameStart : Nat := 67370
def rule : BoundRule := .product (.predecessor 0 67427 .coefficient) (.predecessor 1 67428 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67427 .coefficient)
      LeftAuthority67425.bound (LeftAuthority67425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67428 .coefficient)
      LeftBound67423.bound (LeftBound67423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67423.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority67425.bound LeftBound67423.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67425.bound, LeftBound67423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority67425.actual selector witness) * (LeftBound67423.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67429

namespace LeftBound67445
def owner : Owner := ⟨.program ⟨214⟩, ⟨7871⟩⟩
def transferEvent : Nat := 67445
def frameStart : Nat := 67370
def rule : BoundRule := .scale (.predecessor 0 67443 .coefficient) (.value (.predecessor 1 67444 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67443 .coefficient)
      LeftAuthority67441.bound (LeftAuthority67441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67444 .coefficient)
      LeftAuthority67432.bound (LeftAuthority67432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority67432.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority67441.bound LeftAuthority67432.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67441.bound, LeftAuthority67432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority67441.actual selector witness) * (LeftAuthority67432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound67445

namespace LeftBound67448
def owner : Owner := ⟨.program ⟨214⟩, ⟨6766⟩⟩
def transferEvent : Nat := 67448
def frameStart : Nat := 67370
def rule : BoundRule := .identity (.predecessor 0 67447 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67447 .coefficient)
      LeftAuthority67435.bound (LeftAuthority67435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67435.derived selector witness)

def rawBound : CoeffClass := LeftAuthority67435.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority67435.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound67448

namespace LeftBound67452
def owner : Owner := ⟨.program ⟨214⟩, ⟨7872⟩⟩
def transferEvent : Nat := 67452
def frameStart : Nat := 67370
def rule : BoundRule := .product (.predecessor 0 67450 .coefficient) (.predecessor 1 67451 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67450 .coefficient)
      LeftBound67448.bound (LeftBound67448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67451 .coefficient)
      LeftBound67445.bound (LeftBound67445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67448.bound LeftBound67445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67448.bound, LeftBound67445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67448.actual selector witness) * (LeftBound67445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67452

namespace LeftBound67457
def owner : Owner := ⟨.program ⟨214⟩, ⟨12661⟩⟩
def transferEvent : Nat := 67457
def frameStart : Nat := 67370
def rule : BoundRule := .sum [.predecessor 0 67455 .coefficient, .predecessor 1 67456 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67455 .coefficient)
      LeftBound67452.bound (LeftBound67452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67452.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67456 .coefficient)
      LeftBound67429.bound (LeftBound67429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound67452.bound, LeftBound67429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67452.bound, LeftBound67429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound67452.actual selector witness, LeftBound67429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound67457

namespace LeftBound67461
def owner : Owner := ⟨.program ⟨214⟩, ⟨25448⟩⟩
def transferEvent : Nat := 67461
def frameStart : Nat := 67370
def rule : BoundRule := .product (.predecessor 0 67459 .coefficient) (.predecessor 1 67460 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67459 .coefficient)
      LeftBound67457.bound (LeftBound67457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67457.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67460 .coefficient)
      LeftAuthority67414.bound (LeftAuthority67414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67457.bound LeftAuthority67414.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67457.bound, LeftAuthority67414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67457.actual selector witness) * (LeftAuthority67414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67461

namespace LeftBound67472
def owner : Owner := ⟨.program ⟨214⟩, ⟨16547⟩⟩
def transferEvent : Nat := 67472
def frameStart : Nat := 67370
def rule : BoundRule := .product (.predecessor 0 67470 .coefficient) (.predecessor 1 67471 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 67470 .coefficient)
      LeftAuthority67425.bound (LeftAuthority67425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67425.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 67471 .coefficient)
      LeftAuthority67468.bound (LeftAuthority67468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority67468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority67468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority67425.bound LeftAuthority67468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority67425.bound, LeftAuthority67468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority67425.actual selector witness) * (LeftAuthority67468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound67472

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
