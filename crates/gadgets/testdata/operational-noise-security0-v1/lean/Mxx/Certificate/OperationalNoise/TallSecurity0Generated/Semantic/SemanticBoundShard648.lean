import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard647

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound95285
def owner : Owner := ⟨.program ⟨214⟩, ⟨7105⟩⟩
def transferEvent : Nat := 95285
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95283 .coefficient) (.predecessor 1 95284 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95283 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95284 .coefficient)
      LeftBound7514.bound (LeftBound7514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound7514.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound7514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound7514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95285

namespace LeftBound95290
def owner : Owner := ⟨.program ⟨214⟩, ⟨10122⟩⟩
def transferEvent : Nat := 95290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95288 .coefficient, .predecessor 1 95289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95288 .coefficient)
      LeftBound95285.bound (LeftBound95285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95289 .coefficient)
      LeftBound95280.bound (LeftBound95280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95280.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95285.bound, LeftBound95280.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95285.bound, LeftBound95280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95285.actual selector witness, LeftBound95280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95290

namespace LeftBound95294
def owner : Owner := ⟨.program ⟨214⟩, ⟨10123⟩⟩
def transferEvent : Nat := 95294
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95292 .coefficient, .predecessor 1 95293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95292 .coefficient)
      LeftBound95290.bound (LeftBound95290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95293 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95290.bound, LeftBound7506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95290.bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95290.actual selector witness, LeftBound7506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95294

namespace LeftBound95295
def owner : Owner := ⟨.program ⟨214⟩, ⟨10123⟩⟩
def transferEvent : Nat := 95295
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩ [⟨.result 7507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7507 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨82⟩⟩) (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7506.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95295

namespace LeftBound95300
def owner : Owner := ⟨.program ⟨214⟩, ⟨10124⟩⟩
def transferEvent : Nat := 95300
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95298 .coefficient) (.predecessor 1 95299 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95298 .coefficient)
      LeftBound95294.bound (LeftBound95294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95299 .coefficient)
      LeftBound7503.bound (LeftBound7503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95294.bound LeftBound7503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95294.bound, LeftBound7503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95294.actual selector witness) * (LeftBound7503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95300

namespace LeftBound95301
def owner : Owner := ⟨.program ⟨214⟩, ⟨10124⟩⟩
def transferEvent : Nat := 95301
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩ [⟨.result 7500 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7500 .coefficient)
      LeftAuthority7499.bound (LeftAuthority7499.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7876⟩⟩) (rawTerms := some (Proof.Events029.exact7500RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7499.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7499.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7499.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95301

namespace LeftBound95302
def owner : Owner := ⟨.program ⟨214⟩, ⟨10124⟩⟩
def transferEvent : Nat := 95302
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95297 .summary) (.transfer 95301) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95297 .summary)
      LeftBound95295.bound (LeftBound95295.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10123⟩⟩) (rawTerms := some (Proof.Events372.exact95297RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95301)
      LeftBound95301.bound (LeftBound95301.actual selector witness) := by
  exact .transfer (LeftBound95301.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95295.bound LeftBound95301.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95295.bound, LeftBound95301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95295.actual selector witness) * (LeftBound95301.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95302

namespace LeftBound95310
def owner : Owner := ⟨.program ⟨214⟩, ⟨12941⟩⟩
def transferEvent : Nat := 95310
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 95308 .coefficient, .predecessor 1 95309 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95308 .coefficient)
      LeftBound95300.bound (LeftBound95300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95309 .coefficient)
      LeftBound95272.bound (LeftBound95272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95300.bound, LeftBound95272.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95300.bound, LeftBound95272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95300.actual selector witness, LeftBound95272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95310

namespace LeftBound95312
def owner : Owner := ⟨.program ⟨214⟩, ⟨12941⟩⟩
def transferEvent : Nat := 95312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 95307 .summary, .result 95277 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95307 .summary)
      LeftBound95302.bound (LeftBound95302.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10124⟩⟩) (rawTerms := some (Proof.Events372.exact95307RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95277 .summary)
      LeftBound95274.bound (LeftBound95274.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12940⟩⟩) (rawTerms := some (Proof.Events372.exact95277RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound95302.bound, LeftBound95274.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95302.bound, LeftBound95274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound95302.actual selector witness, LeftBound95274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound95312

namespace LeftBound95316
def owner : Owner := ⟨.program ⟨214⟩, ⟨25592⟩⟩
def transferEvent : Nat := 95316
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95314 .coefficient) (.predecessor 1 95315 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95314 .coefficient)
      LeftBound95310.bound (LeftBound95310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95315 .coefficient)
      LeftAuthority95248.bound (LeftAuthority95248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95248.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95310.bound LeftAuthority95248.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95310.bound, LeftAuthority95248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95310.actual selector witness) * (LeftAuthority95248.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95316

namespace LeftBound95317
def owner : Owner := ⟨.program ⟨214⟩, ⟨25592⟩⟩
def transferEvent : Nat := 95317
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25591⟩⟩]⟩ [⟨.result 95249 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95249 .coefficient)
      LeftAuthority95248.bound (LeftAuthority95248.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25591⟩⟩) (rawTerms := some (Proof.Events372.exact95249RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95248.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95248.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95248.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95317

namespace LeftBound95318
def owner : Owner := ⟨.program ⟨214⟩, ⟨25592⟩⟩
def transferEvent : Nat := 95318
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 95313 .summary) (.transfer 95317) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95313 .summary)
      LeftBound95312.bound (LeftBound95312.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12941⟩⟩) (rawTerms := some (Proof.Events372.exact95313RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound95312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95317)
      LeftBound95317.bound (LeftBound95317.actual selector witness) := by
  exact .transfer (LeftBound95317.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound95312.bound LeftBound95317.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound95312.bound, LeftBound95317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound95312.actual selector witness) * (LeftBound95317.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95318

namespace LeftBound95329
def owner : Owner := ⟨.program ⟨214⟩, ⟨20095⟩⟩
def transferEvent : Nat := 95329
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 95327 .coefficient) (.value (.predecessor 1 95328 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95327 .coefficient)
      LeftAuthority95325.bound (LeftAuthority95325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95328 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority95325.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95325.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95325.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound95329

namespace LeftBound95333
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def transferEvent : Nat := 95333
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 95331 .coefficient) (.predecessor 1 95332 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 95331 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 95332 .coefficient)
      LeftBound95329.bound (LeftBound95329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events372.exact95330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound95329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound95329.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound95329.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound95329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound95329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95333

namespace LeftBound95334
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def transferEvent : Nat := 95334
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20093⟩⟩]⟩ [⟨.result 95326 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 95326 .coefficient)
      LeftAuthority95325.bound (LeftAuthority95325.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20093⟩⟩) (rawTerms := some (Proof.Events372.exact95326RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority95325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority95325.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority95325.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority95325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority95325.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound95334

namespace LeftBound95335
def owner : Owner := ⟨.program ⟨214⟩, ⟨20096⟩⟩
def transferEvent : Nat := 95335
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 95334) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 95334)
      LeftBound95334.bound (LeftBound95334.actual selector witness) := by
  exact .transfer (LeftBound95334.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound95334.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound95334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound95334.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound95335

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
