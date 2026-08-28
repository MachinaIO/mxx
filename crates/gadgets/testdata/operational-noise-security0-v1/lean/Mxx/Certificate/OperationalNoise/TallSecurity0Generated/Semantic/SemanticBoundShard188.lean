import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28259
def owner : Owner := ⟨.program ⟨214⟩, ⟨19254⟩⟩
def transferEvent : Nat := 28259
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 28257 .coefficient) (.value (.predecessor 1 28258 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28257 .coefficient)
      LeftAuthority28255.bound (LeftAuthority28255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28258 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28255.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28255.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28255.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28259

namespace LeftBound28263
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def transferEvent : Nat := 28263
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28261 .coefficient) (.predecessor 1 28262 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28261 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28262 .coefficient)
      LeftBound28259.bound (LeftBound28259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28259.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound28259.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound28259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound28259.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28263

namespace LeftBound28264
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def transferEvent : Nat := 28264
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19252⟩⟩]⟩ [⟨.result 28256 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28256 .coefficient)
      LeftAuthority28255.bound (LeftAuthority28255.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19252⟩⟩) (rawTerms := some (Proof.Events110.exact28256RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28255.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28255.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28255.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28264

namespace LeftBound28265
def owner : Owner := ⟨.program ⟨214⟩, ⟨19255⟩⟩
def transferEvent : Nat := 28265
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 28264) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28264)
      LeftBound28264.bound (LeftBound28264.actual selector witness) := by
  exact .transfer (LeftBound28264.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound28264.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound28264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound28264.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28265

namespace LeftBound28344
def owner : Owner := ⟨.program ⟨214⟩, ⟨12191⟩⟩
def transferEvent : Nat := 28344
def frameStart : Nat := 28315
def rule : BoundRule := .product (.predecessor 0 28342 .coefficient) (.predecessor 1 28343 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28342 .coefficient)
      LeftAuthority28340.bound (LeftAuthority28340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28343 .coefficient)
      LeftAuthority28337.bound (LeftAuthority28337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28337.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28337.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28340.bound LeftAuthority28337.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28340.bound, LeftAuthority28337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28340.actual selector witness) * (LeftAuthority28337.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28344

namespace LeftBound28348
def owner : Owner := ⟨.program ⟨214⟩, ⟨12192⟩⟩
def transferEvent : Nat := 28348
def frameStart : Nat := 28315
def rule : BoundRule := .identity (.predecessor 0 28347 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28347 .coefficient)
      LeftBound28344.bound (LeftBound28344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28344.derived selector witness)

def rawBound : CoeffClass := LeftBound28344.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28344.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28348

namespace LeftBound28365
def owner : Owner := ⟨.program ⟨214⟩, ⟨12282⟩⟩
def transferEvent : Nat := 28365
def frameStart : Nat := 28315
def rule : BoundRule := .sum [.predecessor 0 28363 .coefficient, .predecessor 1 28364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28363 .coefficient)
      LeftBound28348.bound (LeftBound28348.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28364 .coefficient)
      LeftAuthority28361.bound (LeftAuthority28361.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28361.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28348.bound, LeftAuthority28361.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28348.bound, LeftAuthority28361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28348.actual selector witness, LeftAuthority28361.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28365

namespace LeftBound28368
def owner : Owner := ⟨.program ⟨214⟩, ⟨12283⟩⟩
def transferEvent : Nat := 28368
def frameStart : Nat := 28315
def rule : BoundRule := .identity (.predecessor 0 28367 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28367 .coefficient)
      LeftBound28365.bound (LeftBound28365.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28365.derived selector witness)

def rawBound : CoeffClass := LeftBound28365.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28365.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28368

namespace LeftBound28374
def owner : Owner := ⟨.program ⟨214⟩, ⟨12284⟩⟩
def transferEvent : Nat := 28374
def frameStart : Nat := 28315
def rule : BoundRule := .product (.predecessor 0 28372 .coefficient) (.predecessor 1 28373 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28372 .coefficient)
      LeftAuthority28370.bound (LeftAuthority28370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28373 .coefficient)
      LeftBound28368.bound (LeftBound28368.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28368.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28368.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority28370.bound LeftBound28368.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28370.bound, LeftBound28368.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority28370.actual selector witness) * (LeftBound28368.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28374

namespace LeftBound28390
def owner : Owner := ⟨.program ⟨214⟩, ⟨7841⟩⟩
def transferEvent : Nat := 28390
def frameStart : Nat := 28315
def rule : BoundRule := .scale (.predecessor 0 28388 .coefficient) (.value (.predecessor 1 28389 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28388 .coefficient)
      LeftAuthority28386.bound (LeftAuthority28386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28389 .coefficient)
      LeftAuthority28377.bound (LeftAuthority28377.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28377.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28386.bound LeftAuthority28377.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28386.bound, LeftAuthority28377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28386.actual selector witness) * (LeftAuthority28377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28390

namespace LeftBound28393
def owner : Owner := ⟨.program ⟨214⟩, ⟨6792⟩⟩
def transferEvent : Nat := 28393
def frameStart : Nat := 28315
def rule : BoundRule := .identity (.predecessor 0 28392 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28392 .coefficient)
      LeftAuthority28380.bound (LeftAuthority28380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28380.derived selector witness)

def rawBound : CoeffClass := LeftAuthority28380.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority28380.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28393

namespace LeftBound28397
def owner : Owner := ⟨.program ⟨214⟩, ⟨7842⟩⟩
def transferEvent : Nat := 28397
def frameStart : Nat := 28315
def rule : BoundRule := .product (.predecessor 0 28395 .coefficient) (.predecessor 1 28396 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28395 .coefficient)
      LeftBound28393.bound (LeftBound28393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28393.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28396 .coefficient)
      LeftBound28390.bound (LeftBound28390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28390.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28393.bound LeftBound28390.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28393.bound, LeftBound28390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28393.actual selector witness) * (LeftBound28390.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28397

namespace LeftBound28402
def owner : Owner := ⟨.program ⟨214⟩, ⟨12285⟩⟩
def transferEvent : Nat := 28402
def frameStart : Nat := 28315
def rule : BoundRule := .sum [.predecessor 0 28400 .coefficient, .predecessor 1 28401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28400 .coefficient)
      LeftBound28397.bound (LeftBound28397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28401 .coefficient)
      LeftBound28374.bound (LeftBound28374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28397.bound, LeftBound28374.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28397.bound, LeftBound28374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28397.actual selector witness, LeftBound28374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28402

namespace LeftBound28406
def owner : Owner := ⟨.program ⟨214⟩, ⟨25314⟩⟩
def transferEvent : Nat := 28406
def frameStart : Nat := 28315
def rule : BoundRule := .product (.predecessor 0 28404 .coefficient) (.predecessor 1 28405 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28404 .coefficient)
      LeftBound28402.bound (LeftBound28402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28405 .coefficient)
      LeftAuthority28359.bound (LeftAuthority28359.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28359.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28359.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28402.bound LeftAuthority28359.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28402.bound, LeftAuthority28359.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28402.actual selector witness) * (LeftAuthority28359.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28406

namespace LeftBound28417
def owner : Owner := ⟨.program ⟨214⟩, ⟨15436⟩⟩
def transferEvent : Nat := 28417
def frameStart : Nat := 28315
def rule : BoundRule := .product (.predecessor 0 28415 .coefficient) (.predecessor 1 28416 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28415 .coefficient)
      LeftAuthority28370.bound (LeftAuthority28370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28416 .coefficient)
      LeftAuthority28413.bound (LeftAuthority28413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28414RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28413.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28413.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28370.bound LeftAuthority28413.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28370.bound, LeftAuthority28413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28370.actual selector witness) * (LeftAuthority28413.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28417

namespace LeftBound28425
def owner : Owner := ⟨.program ⟨214⟩, ⟨15437⟩⟩
def transferEvent : Nat := 28425
def frameStart : Nat := 28315
def rule : BoundRule := .sum [.predecessor 0 28423 .coefficient, .predecessor 1 28424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28423 .coefficient)
      LeftAuthority28421.bound (LeftAuthority28421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28424 .coefficient)
      LeftBound28417.bound (LeftBound28417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28417.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28421.bound, LeftBound28417.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28421.bound, LeftBound28417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28421.actual selector witness, LeftBound28417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28425

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
