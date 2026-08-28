import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard589
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard590

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86407
def owner : Owner := ⟨.program ⟨214⟩, ⟨15585⟩⟩
def transferEvent : Nat := 86407
def frameStart : Nat := 86307
def rule : BoundRule := .product (.predecessor 0 86405 .coefficient) (.predecessor 1 86406 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86405 .coefficient)
      LeftAuthority86362.bound (LeftAuthority86362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86406 .coefficient)
      LeftAuthority86403.bound (LeftAuthority86403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86403.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86403.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86362.bound LeftAuthority86403.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86362.bound, LeftAuthority86403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86362.actual selector witness) * (LeftAuthority86403.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86407

namespace LeftBound86415
def owner : Owner := ⟨.program ⟨214⟩, ⟨15586⟩⟩
def transferEvent : Nat := 86415
def frameStart : Nat := 86307
def rule : BoundRule := .sum [.predecessor 0 86413 .coefficient, .predecessor 1 86414 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86413 .coefficient)
      LeftAuthority86411.bound (LeftAuthority86411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86414 .coefficient)
      LeftBound86407.bound (LeftBound86407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86407.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86411.bound, LeftBound86407.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86411.bound, LeftBound86407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority86411.actual selector witness, LeftBound86407.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86415

namespace LeftBound86419
def owner : Owner := ⟨.program ⟨214⟩, ⟨25839⟩⟩
def transferEvent : Nat := 86419
def frameStart : Nat := 86307
def rule : BoundRule := .sum [.predecessor 0 86417 .coefficient, .predecessor 1 86418 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86417 .coefficient)
      LeftBound86415.bound (LeftBound86415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86418 .coefficient)
      LeftBound86396.bound (LeftBound86396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86415.bound, LeftBound86396.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86415.bound, LeftBound86396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86415.actual selector witness, LeftBound86396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86419

namespace LeftBound86432
def owner : Owner := ⟨.program ⟨214⟩, ⟨25837⟩⟩
def transferEvent : Nat := 86432
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86430 .coefficient, .predecessor 1 86431 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86430 .coefficient)
      LeftBound86255.bound (LeftBound86255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86431 .coefficient)
      LeftBound86238.bound (LeftBound86238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86255.bound, LeftBound86238.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86255.bound, LeftBound86238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86255.actual selector witness, LeftBound86238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86432

namespace LeftBound86435
def owner : Owner := ⟨.program ⟨214⟩, ⟨25837⟩⟩
def transferEvent : Nat := 86435
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 86429 .summary, .result 86245 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86429 .summary)
      LeftBound86257.bound (LeftBound86257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19315⟩⟩) (rawTerms := some (Proof.Events337.exact86429RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86245 .summary)
      LeftBound86240.bound (LeftBound86240.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25836⟩⟩) (rawTerms := some (Proof.Events336.exact86245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86240.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86257.bound, LeftBound86240.bound]
def bound : CoeffClass := .finite ⟨352036291489792, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86257.bound, LeftBound86240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86257.actual selector witness, LeftBound86240.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86435

namespace LeftBound86439
def owner : Owner := ⟨.program ⟨214⟩, ⟨27217⟩⟩
def transferEvent : Nat := 86439
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86437 .coefficient) (.predecessor 1 86438 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86437 .coefficient)
      LeftBound86432.bound (LeftBound86432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86438 .coefficient)
      LeftAuthority86160.bound (LeftAuthority86160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86160.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86160.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86432.bound LeftAuthority86160.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86432.bound, LeftAuthority86160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86432.actual selector witness) * (LeftAuthority86160.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86439

namespace LeftBound86440
def owner : Owner := ⟨.program ⟨214⟩, ⟨27217⟩⟩
def transferEvent : Nat := 86440
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27215⟩⟩]⟩ [⟨.result 86161 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86161 .coefficient)
      LeftAuthority86160.bound (LeftAuthority86160.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27215⟩⟩) (rawTerms := some (Proof.Events336.exact86161RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86160.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86160.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86160.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86160.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86440

namespace LeftBound86441
def owner : Owner := ⟨.program ⟨214⟩, ⟨27217⟩⟩
def transferEvent : Nat := 86441
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86436 .summary) (.transfer 86440) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86436 .summary)
      LeftBound86435.bound (LeftBound86435.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25837⟩⟩) (rawTerms := some (Proof.Events337.exact86436RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86440)
      LeftBound86440.bound (LeftBound86440.actual selector witness) := by
  exact .transfer (LeftBound86440.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86435.bound LeftBound86440.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86435.bound, LeftBound86440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86435.actual selector witness) * (LeftBound86440.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86441

namespace LeftBound86452
def owner : Owner := ⟨.program ⟨214⟩, ⟨20970⟩⟩
def transferEvent : Nat := 86452
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 86450 .coefficient) (.value (.predecessor 1 86451 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86450 .coefficient)
      LeftAuthority86448.bound (LeftAuthority86448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86451 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority86448.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86448.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86448.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86452

namespace LeftBound86456
def owner : Owner := ⟨.program ⟨214⟩, ⟨20971⟩⟩
def transferEvent : Nat := 86456
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86454 .coefficient) (.predecessor 1 86455 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86454 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86455 .coefficient)
      LeftBound86452.bound (LeftBound86452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86452.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound86452.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound86452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound86452.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86456

namespace LeftBound86457
def owner : Owner := ⟨.program ⟨214⟩, ⟨20971⟩⟩
def transferEvent : Nat := 86457
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20968⟩⟩]⟩ [⟨.result 86449 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86449 .coefficient)
      LeftAuthority86448.bound (LeftAuthority86448.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20968⟩⟩) (rawTerms := some (Proof.Events337.exact86449RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86448.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86448.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86448.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86448.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86457

namespace LeftBound86458
def owner : Owner := ⟨.program ⟨214⟩, ⟨20971⟩⟩
def transferEvent : Nat := 86458
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 86457) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86457)
      LeftBound86457.bound (LeftBound86457.actual selector witness) := by
  exact .transfer (LeftBound86457.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound86457.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound86457.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound86457.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86458

namespace LeftBound86553
def owner : Owner := ⟨.program ⟨214⟩, ⟨15584⟩⟩
def transferEvent : Nat := 86553
def frameStart : Nat := 86514
def rule : BoundRule := .identity (.predecessor 0 86552 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86552 .coefficient)
      LeftAuthority86550.bound (LeftAuthority86550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86550.derived selector witness)

def rawBound : CoeffClass := LeftAuthority86550.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority86550.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86553

namespace LeftBound86570
def owner : Owner := ⟨.program ⟨214⟩, ⟨15658⟩⟩
def transferEvent : Nat := 86570
def frameStart : Nat := 86514
def rule : BoundRule := .sum [.predecessor 0 86568 .coefficient, .predecessor 1 86569 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86568 .coefficient)
      LeftBound86553.bound (LeftBound86553.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86553.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86569 .coefficient)
      LeftAuthority86566.bound (LeftAuthority86566.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86553.bound, LeftAuthority86566.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86553.bound, LeftAuthority86566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86553.actual selector witness, LeftAuthority86566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86570

namespace LeftBound86573
def owner : Owner := ⟨.program ⟨214⟩, ⟨15659⟩⟩
def transferEvent : Nat := 86573
def frameStart : Nat := 86514
def rule : BoundRule := .identity (.predecessor 0 86572 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86572 .coefficient)
      LeftBound86570.bound (LeftBound86570.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86570.derived selector witness)

def rawBound : CoeffClass := LeftBound86570.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86570.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86573

namespace LeftBound86579
def owner : Owner := ⟨.program ⟨214⟩, ⟨15660⟩⟩
def transferEvent : Nat := 86579
def frameStart : Nat := 86514
def rule : BoundRule := .product (.predecessor 0 86577 .coefficient) (.predecessor 1 86578 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86577 .coefficient)
      LeftAuthority86575.bound (LeftAuthority86575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86578 .coefficient)
      LeftBound86573.bound (LeftBound86573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events338.exact86574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority86575.bound LeftBound86573.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86575.bound, LeftBound86573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority86575.actual selector witness) * (LeftBound86573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86579

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
