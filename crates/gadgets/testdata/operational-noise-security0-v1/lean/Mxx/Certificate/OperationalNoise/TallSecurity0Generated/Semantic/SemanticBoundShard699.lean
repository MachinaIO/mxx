import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard092
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard698

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101376
def owner : Owner := ⟨.program ⟨214⟩, ⟨9494⟩⟩
def transferEvent : Nat := 101376
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101374 .coefficient) (.predecessor 1 101375 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101374 .coefficient)
      LeftBound101370.bound (LeftBound101370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101375 .coefficient)
      LeftBound14517.bound (LeftBound14517.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events056.exact14518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14517.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14517.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101370.bound LeftBound14517.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101370.bound, LeftBound14517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101370.actual selector witness) * (LeftBound14517.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101376

namespace LeftBound101377
def owner : Owner := ⟨.program ⟨214⟩, ⟨9494⟩⟩
def transferEvent : Nat := 101377
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7834⟩⟩]⟩ [⟨.result 14514 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 14514 .coefficient)
      LeftAuthority14513.bound (LeftAuthority14513.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7834⟩⟩) (rawTerms := some (Proof.Events056.exact14514RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14513.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14513.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14513.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority14513.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101377

namespace LeftBound101378
def owner : Owner := ⟨.program ⟨214⟩, ⟨9494⟩⟩
def transferEvent : Nat := 101378
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101373 .summary) (.transfer 101377) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101373 .summary)
      LeftBound101371.bound (LeftBound101371.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9493⟩⟩) (rawTerms := some (Proof.Events395.exact101373RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101371.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101377)
      LeftBound101377.bound (LeftBound101377.actual selector witness) := by
  exact .transfer (LeftBound101377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101371.bound LeftBound101377.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101371.bound, LeftBound101377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101371.actual selector witness) * (LeftBound101377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101378

namespace LeftBound101386
def owner : Owner := ⟨.program ⟨214⟩, ⟨10659⟩⟩
def transferEvent : Nat := 101386
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101384 .coefficient, .predecessor 1 101385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101384 .coefficient)
      LeftBound101376.bound (LeftBound101376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101376.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101385 .coefficient)
      LeftBound101348.bound (LeftBound101348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101348.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101376.bound, LeftBound101348.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101376.bound, LeftBound101348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101376.actual selector witness, LeftBound101348.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101386

namespace LeftBound101388
def owner : Owner := ⟨.program ⟨214⟩, ⟨10659⟩⟩
def transferEvent : Nat := 101388
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 101383 .summary, .result 101353 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101383 .summary)
      LeftBound101378.bound (LeftBound101378.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9494⟩⟩) (rawTerms := some (Proof.Events396.exact101383RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101353 .summary)
      LeftBound101350.bound (LeftBound101350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10658⟩⟩) (rawTerms := some (Proof.Events395.exact101353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101350.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101378.bound, LeftBound101350.bound]
def bound : CoeffClass := .finite ⟨95422912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101378.bound, LeftBound101350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101378.actual selector witness, LeftBound101350.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101388

namespace LeftBound101392
def owner : Owner := ⟨.program ⟨214⟩, ⟨24976⟩⟩
def transferEvent : Nat := 101392
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101390 .coefficient) (.predecessor 1 101391 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101390 .coefficient)
      LeftBound101386.bound (LeftBound101386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101391 .coefficient)
      LeftAuthority101324.bound (LeftAuthority101324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101386.bound LeftAuthority101324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101386.bound, LeftAuthority101324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101386.actual selector witness) * (LeftAuthority101324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101392

namespace LeftBound101393
def owner : Owner := ⟨.program ⟨214⟩, ⟨24976⟩⟩
def transferEvent : Nat := 101393
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24975⟩⟩]⟩ [⟨.result 101325 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101325 .coefficient)
      LeftAuthority101324.bound (LeftAuthority101324.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24975⟩⟩) (rawTerms := some (Proof.Events395.exact101325RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101324.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101324.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101324.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101393

namespace LeftBound101394
def owner : Owner := ⟨.program ⟨214⟩, ⟨24976⟩⟩
def transferEvent : Nat := 101394
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101389 .summary) (.transfer 101393) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101389 .summary)
      LeftBound101388.bound (LeftBound101388.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10659⟩⟩) (rawTerms := some (Proof.Events396.exact101389RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101393)
      LeftBound101393.bound (LeftBound101393.actual selector witness) := by
  exact .transfer (LeftBound101393.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101388.bound LeftBound101393.bound
def bound : CoeffClass := .finite ⟨350203613806592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101388.bound, LeftBound101393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101388.actual selector witness) * (LeftBound101393.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101394

namespace LeftBound101405
def owner : Owner := ⟨.program ⟨214⟩, ⟨19087⟩⟩
def transferEvent : Nat := 101405
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 101403 .coefficient) (.value (.predecessor 1 101404 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101403 .coefficient)
      LeftAuthority101401.bound (LeftAuthority101401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101404 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101401.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101401.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101401.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101405

namespace LeftBound101409
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def transferEvent : Nat := 101409
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101407 .coefficient) (.predecessor 1 101408 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101407 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101408 .coefficient)
      LeftBound101405.bound (LeftBound101405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101405.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound101405.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound101405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound101405.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101409

namespace LeftBound101410
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def transferEvent : Nat := 101410
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19085⟩⟩]⟩ [⟨.result 101402 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101402 .coefficient)
      LeftAuthority101401.bound (LeftAuthority101401.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19085⟩⟩) (rawTerms := some (Proof.Events396.exact101402RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101401.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101401.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101401.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101410

namespace LeftBound101411
def owner : Owner := ⟨.program ⟨214⟩, ⟨19088⟩⟩
def transferEvent : Nat := 101411
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 101410) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101410)
      LeftBound101410.bound (LeftBound101410.actual selector witness) := by
  exact .transfer (LeftBound101410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound101410.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound101410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound101410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101411

namespace LeftBound101466
def owner : Owner := ⟨.program ⟨214⟩, ⟨10653⟩⟩
def transferEvent : Nat := 101466
def frameStart : Nat := 101449
def rule : BoundRule := .product (.predecessor 0 101464 .coefficient) (.predecessor 1 101465 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101464 .coefficient)
      LeftAuthority101462.bound (LeftAuthority101462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101462.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101465 .coefficient)
      LeftAuthority101459.bound (LeftAuthority101459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101459.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101462.bound LeftAuthority101459.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101462.bound, LeftAuthority101459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101462.actual selector witness) * (LeftAuthority101459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101466

namespace LeftBound101470
def owner : Owner := ⟨.program ⟨214⟩, ⟨10654⟩⟩
def transferEvent : Nat := 101470
def frameStart : Nat := 101449
def rule : BoundRule := .identity (.predecessor 0 101469 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101469 .coefficient)
      LeftBound101466.bound (LeftBound101466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events396.exact101468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101466.derived selector witness)

def rawBound : CoeffClass := LeftBound101466.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101466.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101466.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101470

namespace LeftBound101487
def owner : Owner := ⟨.program ⟨214⟩, ⟨10764⟩⟩
def transferEvent : Nat := 101487
def frameStart : Nat := 101449
def rule : BoundRule := .sum [.predecessor 0 101485 .coefficient, .predecessor 1 101486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101485 .coefficient)
      LeftBound101470.bound (LeftBound101470.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101486 .coefficient)
      LeftAuthority101483.bound (LeftAuthority101483.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101470.bound, LeftAuthority101483.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101470.bound, LeftAuthority101483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101470.actual selector witness, LeftAuthority101483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101487

namespace LeftBound101490
def owner : Owner := ⟨.program ⟨214⟩, ⟨10765⟩⟩
def transferEvent : Nat := 101490
def frameStart : Nat := 101449
def rule : BoundRule := .identity (.predecessor 0 101489 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101489 .coefficient)
      LeftBound101487.bound (LeftBound101487.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101487.derived selector witness)

def rawBound : CoeffClass := LeftBound101487.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101487.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101487.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101490

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
