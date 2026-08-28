import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard495

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound78246
def owner : Owner := ⟨.program ⟨257⟩, ⟨38811⟩⟩
def transferEvent : Nat := 78246
def frameStart : Nat := 78187
def rule : BoundRule := .identity (.predecessor 0 78245 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78245 .coefficient)
      LeftBound78243.bound (LeftBound78243.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound78243.derived selector witness)

def rawBound : CoeffClass := LeftBound78243.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound78243.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound78246

namespace LeftBound78252
def owner : Owner := ⟨.program ⟨257⟩, ⟨38812⟩⟩
def transferEvent : Nat := 78252
def frameStart : Nat := 78187
def rule : BoundRule := .product (.predecessor 0 78250 .coefficient) (.predecessor 1 78251 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78250 .coefficient)
      LeftAuthority78248.bound (LeftAuthority78248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78251 .coefficient)
      LeftBound78246.bound (LeftBound78246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78246.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority78248.bound LeftBound78246.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78248.bound, LeftBound78246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority78248.actual selector witness) * (LeftBound78246.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78252

namespace LeftBound78260
def owner : Owner := ⟨.program ⟨257⟩, ⟨38813⟩⟩
def transferEvent : Nat := 78260
def frameStart : Nat := 78187
def rule : BoundRule := .sum [.predecessor 0 78258 .coefficient, .predecessor 1 78259 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78258 .coefficient)
      LeftAuthority78256.bound (LeftAuthority78256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78259 .coefficient)
      LeftBound78252.bound (LeftBound78252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78252.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78256.bound, LeftBound78252.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78256.bound, LeftBound78252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority78256.actual selector witness, LeftBound78252.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78260

namespace LeftBound78264
def owner : Owner := ⟨.program ⟨257⟩, ⟨39460⟩⟩
def transferEvent : Nat := 78264
def frameStart : Nat := 78187
def rule : BoundRule := .product (.predecessor 0 78262 .coefficient) (.predecessor 1 78263 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78262 .coefficient)
      LeftBound78260.bound (LeftBound78260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78263 .coefficient)
      LeftAuthority78237.bound (LeftAuthority78237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78237.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound78260.bound LeftAuthority78237.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78260.bound, LeftAuthority78237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound78260.actual selector witness) * (LeftAuthority78237.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78264

namespace LeftBound78275
def owner : Owner := ⟨.program ⟨257⟩, ⟨37722⟩⟩
def transferEvent : Nat := 78275
def frameStart : Nat := 78187
def rule : BoundRule := .product (.predecessor 0 78273 .coefficient) (.predecessor 1 78274 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78273 .coefficient)
      LeftAuthority78248.bound (LeftAuthority78248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78248.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78274 .coefficient)
      LeftAuthority78271.bound (LeftAuthority78271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority78248.bound LeftAuthority78271.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78248.bound, LeftAuthority78271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority78248.actual selector witness) * (LeftAuthority78271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78275

namespace LeftBound78283
def owner : Owner := ⟨.program ⟨257⟩, ⟨37723⟩⟩
def transferEvent : Nat := 78283
def frameStart : Nat := 78187
def rule : BoundRule := .sum [.predecessor 0 78281 .coefficient, .predecessor 1 78282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78281 .coefficient)
      LeftAuthority78279.bound (LeftAuthority78279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority78279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority78279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78282 .coefficient)
      LeftBound78275.bound (LeftBound78275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority78279.bound, LeftBound78275.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority78279.bound, LeftBound78275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority78279.actual selector witness, LeftBound78275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78283

namespace LeftBound78287
def owner : Owner := ⟨.program ⟨257⟩, ⟨39463⟩⟩
def transferEvent : Nat := 78287
def frameStart : Nat := 78187
def rule : BoundRule := .sum [.predecessor 0 78285 .coefficient, .predecessor 1 78286 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78285 .coefficient)
      LeftBound78283.bound (LeftBound78283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78286 .coefficient)
      LeftBound78264.bound (LeftBound78264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78283.bound, LeftBound78264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78283.bound, LeftBound78264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound78283.actual selector witness, LeftBound78264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78287

namespace LeftBound78300
def owner : Owner := ⟨.program ⟨257⟩, ⟨39462⟩⟩
def transferEvent : Nat := 78300
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78298 .coefficient, .predecessor 1 78299 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78298 .coefficient)
      LeftBound78129.bound (LeftBound78129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78299 .coefficient)
      LeftBound78112.bound (LeftBound78112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78112.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78129.bound, LeftBound78112.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78129.bound, LeftBound78112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound78129.actual selector witness, LeftBound78112.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78300

namespace LeftBound78303
def owner : Owner := ⟨.program ⟨257⟩, ⟨39462⟩⟩
def transferEvent : Nat := 78303
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 78297 .summary, .result 78119 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 78297 .summary)
      LeftBound78131.bound (LeftBound78131.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38299⟩⟩) (rawTerms := some (Proof.Events305.exact78297RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 78119 .summary)
      LeftBound78114.bound (LeftBound78114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39461⟩⟩) (rawTerms := some (Proof.Events305.exact78119RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound78114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78131.bound, LeftBound78114.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78131.bound, LeftBound78114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound78131.actual selector witness, LeftBound78114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78303

namespace LeftBound78327
def owner : Owner := ⟨.program ⟨257⟩, ⟨34581⟩⟩
def transferEvent : Nat := 78327
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 78325 .coefficient) (.predecessor 1 78326 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78325 .coefficient)
      LeftAuthority3200.bound (LeftAuthority3200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78326 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3200.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3200.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3200.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound78327

namespace LeftBound78332
def owner : Owner := ⟨.program ⟨257⟩, ⟨10338⟩⟩
def transferEvent : Nat := 78332
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78330 .coefficient) (.predecessor 1 78331 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78330 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78331 .coefficient)
      LeftBound19584.bound (LeftBound19584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19584.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftBound19584.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftBound19584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftBound19584.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78332

namespace LeftBound78337
def owner : Owner := ⟨.program ⟨257⟩, ⟨34582⟩⟩
def transferEvent : Nat := 78337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78335 .coefficient, .predecessor 1 78336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78335 .coefficient)
      LeftBound78332.bound (LeftBound78332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78336 .coefficient)
      LeftBound78327.bound (LeftBound78327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events305.exact78329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78332.bound, LeftBound78327.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78332.bound, LeftBound78327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound78332.actual selector witness, LeftBound78327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78337

namespace LeftBound78341
def owner : Owner := ⟨.program ⟨257⟩, ⟨34583⟩⟩
def transferEvent : Nat := 78341
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 78339 .coefficient, .predecessor 1 78340 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78339 .coefficient)
      LeftBound78337.bound (LeftBound78337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78340 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound78337.bound, LeftBound19576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78337.bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound78337.actual selector witness, LeftBound19576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound78341

namespace LeftBound78342
def owner : Owner := ⟨.program ⟨257⟩, ⟨34583⟩⟩
def transferEvent : Nat := 78342
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨106⟩⟩]⟩ [⟨.result 19577 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19577 .coefficient)
      LeftBound19576.bound (LeftBound19576.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events076.exact19577RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19576.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound19576.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound19576.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78342

namespace LeftBound78347
def owner : Owner := ⟨.program ⟨257⟩, ⟨34584⟩⟩
def transferEvent : Nat := 78347
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 78345 .coefficient) (.predecessor 1 78346 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 78345 .coefficient)
      LeftBound78341.bound (LeftBound78341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events306.exact78344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound78341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound78341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 78346 .coefficient)
      LeftAuthority3203.bound (LeftAuthority3203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3203.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound78341.bound LeftAuthority3203.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound78341.bound, LeftAuthority3203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound78341.actual selector witness) * (LeftAuthority3203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound78347

namespace LeftBound78348
def owner : Owner := ⟨.program ⟨257⟩, ⟨34584⟩⟩
def transferEvent : Nat := 78348
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13671⟩⟩], []⟩ [⟨.result 3204 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 3204 .coefficient)
      LeftAuthority3203.bound (LeftAuthority3203.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13671⟩⟩) (rawTerms := some (Proof.Events012.exact3204RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3203.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3203.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority3203.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound78348

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
