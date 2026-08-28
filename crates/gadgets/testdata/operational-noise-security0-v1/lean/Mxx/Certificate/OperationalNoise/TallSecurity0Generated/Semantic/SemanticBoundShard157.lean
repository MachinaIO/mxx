import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard156

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24225
def owner : Owner := ⟨.program ⟨214⟩, ⟨16478⟩⟩
def transferEvent : Nat := 24225
def frameStart : Nat := 24186
def rule : BoundRule := .identity (.predecessor 0 24224 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24224 .coefficient)
      LeftAuthority24222.bound (LeftAuthority24222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24222.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24222.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority24222.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24225

namespace LeftBound24242
def owner : Owner := ⟨.program ⟨214⟩, ⟨16517⟩⟩
def transferEvent : Nat := 24242
def frameStart : Nat := 24186
def rule : BoundRule := .sum [.predecessor 0 24240 .coefficient, .predecessor 1 24241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24240 .coefficient)
      LeftBound24225.bound (LeftBound24225.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24241 .coefficient)
      LeftAuthority24238.bound (LeftAuthority24238.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority24238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24225.bound, LeftAuthority24238.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24225.bound, LeftAuthority24238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24225.actual selector witness, LeftAuthority24238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24242

namespace LeftBound24245
def owner : Owner := ⟨.program ⟨214⟩, ⟨16518⟩⟩
def transferEvent : Nat := 24245
def frameStart : Nat := 24186
def rule : BoundRule := .identity (.predecessor 0 24244 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24244 .coefficient)
      LeftBound24242.bound (LeftBound24242.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound24242.derived selector witness)

def rawBound : CoeffClass := LeftBound24242.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound24242.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24245

namespace LeftBound24251
def owner : Owner := ⟨.program ⟨214⟩, ⟨16519⟩⟩
def transferEvent : Nat := 24251
def frameStart : Nat := 24186
def rule : BoundRule := .product (.predecessor 0 24249 .coefficient) (.predecessor 1 24250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24249 .coefficient)
      LeftAuthority24247.bound (LeftAuthority24247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24250 .coefficient)
      LeftBound24245.bound (LeftBound24245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24245.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority24247.bound LeftBound24245.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24247.bound, LeftBound24245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority24247.actual selector witness) * (LeftBound24245.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24251

namespace LeftBound24259
def owner : Owner := ⟨.program ⟨214⟩, ⟨16520⟩⟩
def transferEvent : Nat := 24259
def frameStart : Nat := 24186
def rule : BoundRule := .sum [.predecessor 0 24257 .coefficient, .predecessor 1 24258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24257 .coefficient)
      LeftAuthority24255.bound (LeftAuthority24255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24255.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24258 .coefficient)
      LeftBound24251.bound (LeftBound24251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24251.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24255.bound, LeftBound24251.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24255.bound, LeftBound24251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24255.actual selector witness, LeftBound24251.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24259

namespace LeftBound24263
def owner : Owner := ⟨.program ⟨214⟩, ⟨28991⟩⟩
def transferEvent : Nat := 24263
def frameStart : Nat := 24186
def rule : BoundRule := .product (.predecessor 0 24261 .coefficient) (.predecessor 1 24262 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24261 .coefficient)
      LeftBound24259.bound (LeftBound24259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24262 .coefficient)
      LeftAuthority24236.bound (LeftAuthority24236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24236.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24259.bound LeftAuthority24236.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24259.bound, LeftAuthority24236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24259.actual selector witness) * (LeftAuthority24236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24263

namespace LeftBound24274
def owner : Owner := ⟨.program ⟨214⟩, ⟨17914⟩⟩
def transferEvent : Nat := 24274
def frameStart : Nat := 24186
def rule : BoundRule := .product (.predecessor 0 24272 .coefficient) (.predecessor 1 24273 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24272 .coefficient)
      LeftAuthority24247.bound (LeftAuthority24247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24273 .coefficient)
      LeftAuthority24270.bound (LeftAuthority24270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24270.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24247.bound LeftAuthority24270.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24247.bound, LeftAuthority24270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24247.actual selector witness) * (LeftAuthority24270.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24274

namespace LeftBound24282
def owner : Owner := ⟨.program ⟨214⟩, ⟨17915⟩⟩
def transferEvent : Nat := 24282
def frameStart : Nat := 24186
def rule : BoundRule := .sum [.predecessor 0 24280 .coefficient, .predecessor 1 24281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24280 .coefficient)
      LeftAuthority24278.bound (LeftAuthority24278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24278.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24281 .coefficient)
      LeftBound24274.bound (LeftBound24274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24278.bound, LeftBound24274.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24278.bound, LeftBound24274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24278.actual selector witness, LeftBound24274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24282

namespace LeftBound24286
def owner : Owner := ⟨.program ⟨214⟩, ⟨28995⟩⟩
def transferEvent : Nat := 24286
def frameStart : Nat := 24186
def rule : BoundRule := .sum [.predecessor 0 24284 .coefficient, .predecessor 1 24285 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24284 .coefficient)
      LeftBound24282.bound (LeftBound24282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24285 .coefficient)
      LeftBound24263.bound (LeftBound24263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24263.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24282.bound, LeftBound24263.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24282.bound, LeftBound24263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24282.actual selector witness, LeftBound24263.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24286

namespace LeftBound24299
def owner : Owner := ⟨.program ⟨214⟩, ⟨28993⟩⟩
def transferEvent : Nat := 24299
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24297 .coefficient, .predecessor 1 24298 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24297 .coefficient)
      LeftBound24128.bound (LeftBound24128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24298 .coefficient)
      LeftBound24111.bound (LeftBound24111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24111.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24128.bound, LeftBound24111.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24128.bound, LeftBound24111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24128.actual selector witness, LeftBound24111.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24299

namespace LeftBound24302
def owner : Owner := ⟨.program ⟨214⟩, ⟨28993⟩⟩
def transferEvent : Nat := 24302
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24296 .summary, .result 24118 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24296 .summary)
      LeftBound24130.bound (LeftBound24130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22135⟩⟩) (rawTerms := some (Proof.Events094.exact24296RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24118 .summary)
      LeftBound24113.bound (LeftBound24113.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28992⟩⟩) (rawTerms := some (Proof.Events094.exact24118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24113.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24130.bound, LeftBound24113.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24130.bound, LeftBound24113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24130.actual selector witness, LeftBound24113.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24302

namespace LeftBound24326
def owner : Owner := ⟨.program ⟨214⟩, ⟨11984⟩⟩
def transferEvent : Nat := 24326
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 24324 .coefficient) (.predecessor 1 24325 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24324 .coefficient)
      LeftAuthority979.bound (LeftAuthority979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24325 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority979.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority979.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority979.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24326

namespace LeftBound24331
def owner : Owner := ⟨.program ⟨214⟩, ⟨7354⟩⟩
def transferEvent : Nat := 24331
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24329 .coefficient) (.predecessor 1 24330 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24329 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24330 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24331

namespace LeftBound24336
def owner : Owner := ⟨.program ⟨214⟩, ⟨11985⟩⟩
def transferEvent : Nat := 24336
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24334 .coefficient, .predecessor 1 24335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24334 .coefficient)
      LeftBound24331.bound (LeftBound24331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24335 .coefficient)
      LeftBound24326.bound (LeftBound24326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24331.bound, LeftBound24326.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24331.bound, LeftBound24326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24331.actual selector witness, LeftBound24326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24336

namespace LeftBound24340
def owner : Owner := ⟨.program ⟨214⟩, ⟨11986⟩⟩
def transferEvent : Nat := 24340
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24338 .coefficient, .predecessor 1 24339 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24338 .coefficient)
      LeftBound24336.bound (LeftBound24336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24339 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24336.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24336.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24336.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24340

namespace LeftBound24341
def owner : Owner := ⟨.program ⟨214⟩, ⟨11986⟩⟩
def transferEvent : Nat := 24341
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24341

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
