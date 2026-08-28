import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard463

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound68416
def owner : Owner := ⟨.program ⟨214⟩, ⟨7866⟩⟩
def transferEvent : Nat := 68416
def frameStart : Nat := 68334
def rule : BoundRule := .product (.predecessor 0 68414 .coefficient) (.predecessor 1 68415 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68414 .coefficient)
      LeftBound68412.bound (LeftBound68412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68415 .coefficient)
      LeftBound68409.bound (LeftBound68409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68412.bound LeftBound68409.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68412.bound, LeftBound68409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68412.actual selector witness) * (LeftBound68409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68416

namespace LeftBound68421
def owner : Owner := ⟨.program ⟨214⟩, ⟨12052⟩⟩
def transferEvent : Nat := 68421
def frameStart : Nat := 68334
def rule : BoundRule := .sum [.predecessor 0 68419 .coefficient, .predecessor 1 68420 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68419 .coefficient)
      LeftBound68416.bound (LeftBound68416.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68416.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68420 .coefficient)
      LeftBound68393.bound (LeftBound68393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68416.bound, LeftBound68393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68416.bound, LeftBound68393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68416.actual selector witness, LeftBound68393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68421

namespace LeftBound68425
def owner : Owner := ⟨.program ⟨214⟩, ⟨25217⟩⟩
def transferEvent : Nat := 68425
def frameStart : Nat := 68334
def rule : BoundRule := .product (.predecessor 0 68423 .coefficient) (.predecessor 1 68424 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68423 .coefficient)
      LeftBound68421.bound (LeftBound68421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68424 .coefficient)
      LeftAuthority68378.bound (LeftAuthority68378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68378.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68421.bound LeftAuthority68378.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68421.bound, LeftAuthority68378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68421.actual selector witness) * (LeftAuthority68378.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68425

namespace LeftBound68436
def owner : Owner := ⟨.program ⟨214⟩, ⟨16379⟩⟩
def transferEvent : Nat := 68436
def frameStart : Nat := 68334
def rule : BoundRule := .product (.predecessor 0 68434 .coefficient) (.predecessor 1 68435 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68434 .coefficient)
      LeftAuthority68389.bound (LeftAuthority68389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68389.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68435 .coefficient)
      LeftAuthority68432.bound (LeftAuthority68432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority68389.bound LeftAuthority68432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68389.bound, LeftAuthority68432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority68389.actual selector witness) * (LeftAuthority68432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68436

namespace LeftBound68444
def owner : Owner := ⟨.program ⟨214⟩, ⟨16380⟩⟩
def transferEvent : Nat := 68444
def frameStart : Nat := 68334
def rule : BoundRule := .sum [.predecessor 0 68442 .coefficient, .predecessor 1 68443 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68442 .coefficient)
      LeftAuthority68440.bound (LeftAuthority68440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68443 .coefficient)
      LeftBound68436.bound (LeftBound68436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority68440.bound, LeftBound68436.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68440.bound, LeftBound68436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority68440.actual selector witness, LeftBound68436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68444

namespace LeftBound68448
def owner : Owner := ⟨.program ⟨214⟩, ⟨25218⟩⟩
def transferEvent : Nat := 68448
def frameStart : Nat := 68334
def rule : BoundRule := .sum [.predecessor 0 68446 .coefficient, .predecessor 1 68447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68446 .coefficient)
      LeftBound68444.bound (LeftBound68444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68447 .coefficient)
      LeftBound68425.bound (LeftBound68425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68444.bound, LeftBound68425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68444.bound, LeftBound68425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68444.actual selector witness, LeftBound68425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68448

namespace LeftBound68461
def owner : Owner := ⟨.program ⟨214⟩, ⟨25216⟩⟩
def transferEvent : Nat := 68461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 68459 .coefficient, .predecessor 1 68460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68459 .coefficient)
      LeftBound68282.bound (LeftBound68282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68460 .coefficient)
      LeftBound68265.bound (LeftBound68265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68282.bound, LeftBound68265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68282.bound, LeftBound68265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68282.actual selector witness, LeftBound68265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68461

namespace LeftBound68464
def owner : Owner := ⟨.program ⟨214⟩, ⟨25216⟩⟩
def transferEvent : Nat := 68464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 68458 .summary, .result 68272 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68458 .summary)
      LeftBound68284.bound (LeftBound68284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19815⟩⟩) (rawTerms := some (Proof.Events267.exact68458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68284.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68272 .summary)
      LeftBound68267.bound (LeftBound68267.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25215⟩⟩) (rawTerms := some (Proof.Events266.exact68272RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68267.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound68284.bound, LeftBound68267.bound]
def bound : CoeffClass := .finite ⟨352115681275904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68284.bound, LeftBound68267.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound68284.actual selector witness, LeftBound68267.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound68464

namespace LeftBound68468
def owner : Owner := ⟨.program ⟨214⟩, ⟨28723⟩⟩
def transferEvent : Nat := 68468
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68466 .coefficient) (.predecessor 1 68467 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68466 .coefficient)
      LeftBound68461.bound (LeftBound68461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68467 .coefficient)
      LeftAuthority68187.bound (LeftAuthority68187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events266.exact68188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68187.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68461.bound LeftAuthority68187.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68461.bound, LeftAuthority68187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68461.actual selector witness) * (LeftAuthority68187.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68468

namespace LeftBound68469
def owner : Owner := ⟨.program ⟨214⟩, ⟨28723⟩⟩
def transferEvent : Nat := 68469
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28721⟩⟩]⟩ [⟨.result 68188 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68188 .coefficient)
      LeftAuthority68187.bound (LeftAuthority68187.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28721⟩⟩) (rawTerms := some (Proof.Events266.exact68188RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68187.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68187.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68187.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68469

namespace LeftBound68470
def owner : Owner := ⟨.program ⟨214⟩, ⟨28723⟩⟩
def transferEvent : Nat := 68470
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 68465 .summary) (.transfer 68469) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68465 .summary)
      LeftBound68464.bound (LeftBound68464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25216⟩⟩) (rawTerms := some (Proof.Events267.exact68465RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68469)
      LeftBound68469.bound (LeftBound68469.actual selector witness) := by
  exact .transfer (LeftBound68469.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound68464.bound LeftBound68469.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound68464.bound, LeftBound68469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound68464.actual selector witness) * (LeftBound68469.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68470

namespace LeftBound68481
def owner : Owner := ⟨.program ⟨214⟩, ⟨21974⟩⟩
def transferEvent : Nat := 68481
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 68479 .coefficient) (.value (.predecessor 1 68480 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68479 .coefficient)
      LeftAuthority68477.bound (LeftAuthority68477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68480 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority68477.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68477.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68477.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound68481

namespace LeftBound68485
def owner : Owner := ⟨.program ⟨214⟩, ⟨21975⟩⟩
def transferEvent : Nat := 68485
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 68483 .coefficient) (.predecessor 1 68484 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68483 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 68484 .coefficient)
      LeftBound68481.bound (LeftBound68481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68481.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound68481.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound68481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound68481.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68485

namespace LeftBound68486
def owner : Owner := ⟨.program ⟨214⟩, ⟨21975⟩⟩
def transferEvent : Nat := 68486
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21972⟩⟩]⟩ [⟨.result 68478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 68478 .coefficient)
      LeftAuthority68477.bound (LeftAuthority68477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21972⟩⟩) (rawTerms := some (Proof.Events267.exact68478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority68477.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority68477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound68486

namespace LeftBound68487
def owner : Owner := ⟨.program ⟨214⟩, ⟨21975⟩⟩
def transferEvent : Nat := 68487
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 68486) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 68486)
      LeftBound68486.bound (LeftBound68486.actual selector witness) := by
  exact .transfer (LeftBound68486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound68486.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound68486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound68486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound68487

namespace LeftBound68582
def owner : Owner := ⟨.program ⟨214⟩, ⟨16378⟩⟩
def transferEvent : Nat := 68582
def frameStart : Nat := 68543
def rule : BoundRule := .identity (.predecessor 0 68581 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 68581 .coefficient)
      LeftAuthority68579.bound (LeftAuthority68579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority68579.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority68579.derived selector witness)

def rawBound : CoeffClass := LeftAuthority68579.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority68579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority68579.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound68582

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
