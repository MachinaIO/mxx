import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1346

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound200427
def owner : Owner := ⟨.program ⟨257⟩, ⟨22718⟩⟩
def transferEvent : Nat := 200427
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 200425 .coefficient) (.value (.predecessor 1 200426 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200425 .coefficient)
      LeftAuthority200423.bound (LeftAuthority200423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200424RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200423.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200426 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority200423.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200423.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority200423.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound200427

namespace LeftBound200431
def owner : Owner := ⟨.program ⟨257⟩, ⟨22719⟩⟩
def transferEvent : Nat := 200431
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 200429 .coefficient) (.predecessor 1 200430 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200429 .coefficient)
      LeftBound192992.bound (LeftBound192992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200430 .coefficient)
      LeftBound200427.bound (LeftBound200427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200427.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200427.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192992.bound LeftBound200427.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192992.bound, LeftBound200427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192992.actual selector witness) * (LeftBound200427.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200431

namespace LeftBound200432
def owner : Owner := ⟨.program ⟨257⟩, ⟨22719⟩⟩
def transferEvent : Nat := 200432
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22716⟩⟩]⟩ [⟨.result 200424 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200424 .coefficient)
      LeftAuthority200423.bound (LeftAuthority200423.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22716⟩⟩) (rawTerms := some (Proof.Events782.exact200424RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200423.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200423.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority200423.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority200423.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound200432

namespace LeftBound200433
def owner : Owner := ⟨.program ⟨257⟩, ⟨22719⟩⟩
def transferEvent : Nat := 200433
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 200432) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 200432)
      LeftBound200432.bound (LeftBound200432.actual selector witness) := by
  exact .transfer (LeftBound200432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound200432.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound200432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound200432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200433

namespace LeftBound200528
def owner : Owner := ⟨.program ⟨257⟩, ⟨21825⟩⟩
def transferEvent : Nat := 200528
def frameStart : Nat := 200489
def rule : BoundRule := .identity (.predecessor 0 200527 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200527 .coefficient)
      LeftAuthority200525.bound (LeftAuthority200525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200525.derived selector witness)

def rawBound : CoeffClass := LeftAuthority200525.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority200525.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound200528

namespace LeftBound200545
def owner : Owner := ⟨.program ⟨257⟩, ⟨23294⟩⟩
def transferEvent : Nat := 200545
def frameStart : Nat := 200489
def rule : BoundRule := .sum [.predecessor 0 200543 .coefficient, .predecessor 1 200544 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200543 .coefficient)
      LeftBound200528.bound (LeftBound200528.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound200528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200544 .coefficient)
      LeftAuthority200541.bound (LeftAuthority200541.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority200541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200528.bound, LeftAuthority200541.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200528.bound, LeftAuthority200541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200528.actual selector witness, LeftAuthority200541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200545

namespace LeftBound200548
def owner : Owner := ⟨.program ⟨257⟩, ⟨23295⟩⟩
def transferEvent : Nat := 200548
def frameStart : Nat := 200489
def rule : BoundRule := .identity (.predecessor 0 200547 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200547 .coefficient)
      LeftBound200545.bound (LeftBound200545.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound200545.derived selector witness)

def rawBound : CoeffClass := LeftBound200545.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound200545.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound200548

namespace LeftBound200554
def owner : Owner := ⟨.program ⟨257⟩, ⟨23296⟩⟩
def transferEvent : Nat := 200554
def frameStart : Nat := 200489
def rule : BoundRule := .product (.predecessor 0 200552 .coefficient) (.predecessor 1 200553 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200552 .coefficient)
      LeftAuthority200550.bound (LeftAuthority200550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200553 .coefficient)
      LeftBound200548.bound (LeftBound200548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200548.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority200550.bound LeftBound200548.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200550.bound, LeftBound200548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority200550.actual selector witness) * (LeftBound200548.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200554

namespace LeftBound200562
def owner : Owner := ⟨.program ⟨257⟩, ⟨23297⟩⟩
def transferEvent : Nat := 200562
def frameStart : Nat := 200489
def rule : BoundRule := .sum [.predecessor 0 200560 .coefficient, .predecessor 1 200561 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200560 .coefficient)
      LeftAuthority200558.bound (LeftAuthority200558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200561 .coefficient)
      LeftBound200554.bound (LeftBound200554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority200558.bound, LeftBound200554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200558.bound, LeftBound200554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority200558.actual selector witness, LeftBound200554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200562

namespace LeftBound200566
def owner : Owner := ⟨.program ⟨257⟩, ⟨23935⟩⟩
def transferEvent : Nat := 200566
def frameStart : Nat := 200489
def rule : BoundRule := .product (.predecessor 0 200564 .coefficient) (.predecessor 1 200565 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200564 .coefficient)
      LeftBound200562.bound (LeftBound200562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200565 .coefficient)
      LeftAuthority200539.bound (LeftAuthority200539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200539.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound200562.bound LeftAuthority200539.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200562.bound, LeftAuthority200539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound200562.actual selector witness) * (LeftAuthority200539.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200566

namespace LeftBound200577
def owner : Owner := ⟨.program ⟨257⟩, ⟨22126⟩⟩
def transferEvent : Nat := 200577
def frameStart : Nat := 200489
def rule : BoundRule := .product (.predecessor 0 200575 .coefficient) (.predecessor 1 200576 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200575 .coefficient)
      LeftAuthority200550.bound (LeftAuthority200550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200550.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200576 .coefficient)
      LeftAuthority200573.bound (LeftAuthority200573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200574RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200573.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200573.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority200550.bound LeftAuthority200573.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200550.bound, LeftAuthority200573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority200550.actual selector witness) * (LeftAuthority200573.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound200577

namespace LeftBound200585
def owner : Owner := ⟨.program ⟨257⟩, ⟨22127⟩⟩
def transferEvent : Nat := 200585
def frameStart : Nat := 200489
def rule : BoundRule := .sum [.predecessor 0 200583 .coefficient, .predecessor 1 200584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200583 .coefficient)
      LeftAuthority200581.bound (LeftAuthority200581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority200581.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority200581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200584 .coefficient)
      LeftBound200577.bound (LeftBound200577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200577.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200577.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority200581.bound, LeftBound200577.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority200581.bound, LeftBound200577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority200581.actual selector witness, LeftBound200577.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200585

namespace LeftBound200589
def owner : Owner := ⟨.program ⟨257⟩, ⟨23939⟩⟩
def transferEvent : Nat := 200589
def frameStart : Nat := 200489
def rule : BoundRule := .sum [.predecessor 0 200587 .coefficient, .predecessor 1 200588 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200587 .coefficient)
      LeftBound200585.bound (LeftBound200585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200585.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200588 .coefficient)
      LeftBound200566.bound (LeftBound200566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200585.bound, LeftBound200566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200585.bound, LeftBound200566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200585.actual selector witness, LeftBound200566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200589

namespace LeftBound200602
def owner : Owner := ⟨.program ⟨257⟩, ⟨23937⟩⟩
def transferEvent : Nat := 200602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 200600 .coefficient, .predecessor 1 200601 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200600 .coefficient)
      LeftBound200431.bound (LeftBound200431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events783.exact200599RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200601 .coefficient)
      LeftBound200414.bound (LeftBound200414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events782.exact200421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound200414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound200414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200431.bound, LeftBound200414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200431.bound, LeftBound200414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200431.actual selector witness, LeftBound200414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200602

namespace LeftBound200605
def owner : Owner := ⟨.program ⟨257⟩, ⟨23937⟩⟩
def transferEvent : Nat := 200605
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 200599 .summary, .result 200421 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200599 .summary)
      LeftBound200433.bound (LeftBound200433.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22719⟩⟩) (rawTerms := some (Proof.Events783.exact200599RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 200421 .summary)
      LeftBound200416.bound (LeftBound200416.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23936⟩⟩) (rawTerms := some (Proof.Events782.exact200421RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound200416.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound200433.bound, LeftBound200416.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound200433.bound, LeftBound200416.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound200433.actual selector witness, LeftBound200416.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound200605

namespace LeftBound200629
def owner : Owner := ⟨.program ⟨257⟩, ⟨18325⟩⟩
def transferEvent : Nat := 200629
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 200627 .coefficient) (.predecessor 1 200628 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 200627 .coefficient)
      LeftAuthority9437.bound (LeftAuthority9437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 200628 .coefficient)
      LeftBound192901.bound (LeftBound192901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9437.bound LeftBound192901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9437.bound, LeftBound192901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9437.actual selector witness) * (LeftBound192901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound200629

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
