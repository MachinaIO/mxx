import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard040
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard551

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81293
def owner : Owner := ⟨.program ⟨214⟩, ⟨16828⟩⟩
def transferEvent : Nat := 81293
def frameStart : Nat := 81234
def rule : BoundRule := .identity (.predecessor 0 81292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81292 .coefficient)
      LeftBound81290.bound (LeftBound81290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81290.derived selector witness)

def rawBound : CoeffClass := LeftBound81290.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81293

namespace LeftBound81299
def owner : Owner := ⟨.program ⟨214⟩, ⟨16829⟩⟩
def transferEvent : Nat := 81299
def frameStart : Nat := 81234
def rule : BoundRule := .product (.predecessor 0 81297 .coefficient) (.predecessor 1 81298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81297 .coefficient)
      LeftAuthority81295.bound (LeftAuthority81295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81298 .coefficient)
      LeftBound81293.bound (LeftBound81293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority81295.bound LeftBound81293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81295.bound, LeftBound81293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority81295.actual selector witness) * (LeftBound81293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81299

namespace LeftBound81307
def owner : Owner := ⟨.program ⟨214⟩, ⟨16830⟩⟩
def transferEvent : Nat := 81307
def frameStart : Nat := 81234
def rule : BoundRule := .sum [.predecessor 0 81305 .coefficient, .predecessor 1 81306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81305 .coefficient)
      LeftAuthority81303.bound (LeftAuthority81303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81303.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81306 .coefficient)
      LeftBound81299.bound (LeftBound81299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81303.bound, LeftBound81299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81303.bound, LeftBound81299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81303.actual selector witness, LeftBound81299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81307

namespace LeftBound81311
def owner : Owner := ⟨.program ⟨214⟩, ⟨29603⟩⟩
def transferEvent : Nat := 81311
def frameStart : Nat := 81234
def rule : BoundRule := .product (.predecessor 0 81309 .coefficient) (.predecessor 1 81310 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81309 .coefficient)
      LeftBound81307.bound (LeftBound81307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81310 .coefficient)
      LeftAuthority81284.bound (LeftAuthority81284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81307.bound LeftAuthority81284.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81307.bound, LeftAuthority81284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81307.actual selector witness) * (LeftAuthority81284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81311

namespace LeftBound81322
def owner : Owner := ⟨.program ⟨214⟩, ⟨16799⟩⟩
def transferEvent : Nat := 81322
def frameStart : Nat := 81234
def rule : BoundRule := .product (.predecessor 0 81320 .coefficient) (.predecessor 1 81321 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81320 .coefficient)
      LeftAuthority81295.bound (LeftAuthority81295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81321 .coefficient)
      LeftAuthority81318.bound (LeftAuthority81318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81295.bound LeftAuthority81318.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81295.bound, LeftAuthority81318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81295.actual selector witness) * (LeftAuthority81318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81322

namespace LeftBound81330
def owner : Owner := ⟨.program ⟨214⟩, ⟨16800⟩⟩
def transferEvent : Nat := 81330
def frameStart : Nat := 81234
def rule : BoundRule := .sum [.predecessor 0 81328 .coefficient, .predecessor 1 81329 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81328 .coefficient)
      LeftAuthority81326.bound (LeftAuthority81326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81329 .coefficient)
      LeftBound81322.bound (LeftBound81322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81326.bound, LeftBound81322.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81326.bound, LeftBound81322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81326.actual selector witness, LeftBound81322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81330

namespace LeftBound81334
def owner : Owner := ⟨.program ⟨214⟩, ⟨29607⟩⟩
def transferEvent : Nat := 81334
def frameStart : Nat := 81234
def rule : BoundRule := .sum [.predecessor 0 81332 .coefficient, .predecessor 1 81333 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81332 .coefficient)
      LeftBound81330.bound (LeftBound81330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81333 .coefficient)
      LeftBound81311.bound (LeftBound81311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81330.bound, LeftBound81311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81330.bound, LeftBound81311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81330.actual selector witness, LeftBound81311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81334

namespace LeftBound81347
def owner : Owner := ⟨.program ⟨214⟩, ⟨29605⟩⟩
def transferEvent : Nat := 81347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81345 .coefficient, .predecessor 1 81346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81345 .coefficient)
      LeftBound81176.bound (LeftBound81176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81346 .coefficient)
      LeftBound81159.bound (LeftBound81159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81176.bound, LeftBound81159.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81176.bound, LeftBound81159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81176.actual selector witness, LeftBound81159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81347

namespace LeftBound81350
def owner : Owner := ⟨.program ⟨214⟩, ⟨29605⟩⟩
def transferEvent : Nat := 81350
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81344 .summary, .result 81166 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81344 .summary)
      LeftBound81178.bound (LeftBound81178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22555⟩⟩) (rawTerms := some (Proof.Events317.exact81344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81166 .summary)
      LeftBound81161.bound (LeftBound81161.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29604⟩⟩) (rawTerms := some (Proof.Events317.exact81166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81178.bound, LeftBound81161.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81178.bound, LeftBound81161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81178.actual selector witness, LeftBound81161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81350

namespace LeftBound81374
def owner : Owner := ⟨.program ⟨214⟩, ⟨12765⟩⟩
def transferEvent : Nat := 81374
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 81372 .coefficient) (.predecessor 1 81373 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81372 .coefficient)
      LeftAuthority3896.bound (LeftAuthority3896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81373 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3896.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3896.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3896.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81374

namespace LeftBound81379
def owner : Owner := ⟨.program ⟨214⟩, ⟨7243⟩⟩
def transferEvent : Nat := 81379
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81377 .coefficient) (.predecessor 1 81378 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81377 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81378 .coefficient)
      LeftBound7974.bound (LeftBound7974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7974.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound7974.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound7974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound7974.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81379

namespace LeftBound81384
def owner : Owner := ⟨.program ⟨214⟩, ⟨12766⟩⟩
def transferEvent : Nat := 81384
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81382 .coefficient, .predecessor 1 81383 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81382 .coefficient)
      LeftBound81379.bound (LeftBound81379.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81379.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81379.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81383 .coefficient)
      LeftBound81374.bound (LeftBound81374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81376RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81374.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81374.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81379.bound, LeftBound81374.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81379.bound, LeftBound81374.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81379.actual selector witness, LeftBound81374.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81384

namespace LeftBound81388
def owner : Owner := ⟨.program ⟨214⟩, ⟨12767⟩⟩
def transferEvent : Nat := 81388
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81386 .coefficient, .predecessor 1 81387 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81386 .coefficient)
      LeftBound81384.bound (LeftBound81384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81387 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81384.bound, LeftBound7966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81384.bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81384.actual selector witness, LeftBound7966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81388

namespace LeftBound81389
def owner : Owner := ⟨.program ⟨214⟩, ⟨12767⟩⟩
def transferEvent : Nat := 81389
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨101⟩⟩]⟩ [⟨.result 7967 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7967 .coefficient)
      LeftBound7966.bound (LeftBound7966.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events031.exact7967RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7966.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7966.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7966.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81389

namespace LeftBound81394
def owner : Owner := ⟨.program ⟨214⟩, ⟨12768⟩⟩
def transferEvent : Nat := 81394
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81392 .coefficient) (.predecessor 1 81393 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81392 .coefficient)
      LeftBound81388.bound (LeftBound81388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81391RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81388.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81393 .coefficient)
      LeftAuthority3899.bound (LeftAuthority3899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact3900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound81388.bound LeftAuthority3899.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81388.bound, LeftAuthority3899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound81388.actual selector witness) * (LeftAuthority3899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81394

namespace LeftBound81395
def owner : Owner := ⟨.program ⟨214⟩, ⟨12768⟩⟩
def transferEvent : Nat := 81395
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10030⟩⟩], []⟩ [⟨.result 3900 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3900 .coefficient)
      LeftAuthority3899.bound (LeftAuthority3899.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10030⟩⟩) (rawTerms := some (Proof.Events015.exact3900RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3899.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3899.bound []
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3899.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81395

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
