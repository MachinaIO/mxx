import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1798
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1806

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound267275
def owner : Owner := ⟨.program ⟨257⟩, ⟨44464⟩⟩
def transferEvent : Nat := 267275
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 267270 .summary) (.transfer 267274) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267270 .summary)
      LeftBound267269.bound (LeftBound267269.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44210⟩⟩) (rawTerms := some (Proof.Events1044.exact267270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound267269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 267274)
      LeftBound267274.bound (LeftBound267274.actual selector witness) := by
  exact .transfer (LeftBound267274.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound267269.bound LeftBound267274.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267269.bound, LeftBound267274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound267269.actual selector witness) * (LeftBound267274.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267275

namespace LeftBound267286
def owner : Owner := ⟨.program ⟨257⟩, ⟨43372⟩⟩
def transferEvent : Nat := 267286
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 267284 .coefficient) (.value (.predecessor 1 267285 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267284 .coefficient)
      LeftAuthority267282.bound (LeftAuthority267282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267285 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority267282.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267282.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority267282.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound267286

namespace LeftBound267290
def owner : Owner := ⟨.program ⟨257⟩, ⟨43373⟩⟩
def transferEvent : Nat := 267290
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 267288 .coefficient) (.predecessor 1 267289 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267288 .coefficient)
      LeftBound266117.bound (LeftBound266117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267289 .coefficient)
      LeftBound267286.bound (LeftBound267286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267286.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266117.bound LeftBound267286.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266117.bound, LeftBound267286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266117.actual selector witness) * (LeftBound267286.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267290

namespace LeftBound267291
def owner : Owner := ⟨.program ⟨257⟩, ⟨43373⟩⟩
def transferEvent : Nat := 267291
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨43370⟩⟩]⟩ [⟨.result 267283 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267283 .coefficient)
      LeftAuthority267282.bound (LeftAuthority267282.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨43370⟩⟩) (rawTerms := some (Proof.Events1044.exact267283RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267282.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267282.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority267282.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority267282.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound267291

namespace LeftBound267292
def owner : Owner := ⟨.program ⟨257⟩, ⟨43373⟩⟩
def transferEvent : Nat := 267292
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 266120 .summary) (.transfer 267291) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 266120 .summary)
      LeftBound266118.bound (LeftBound266118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5449⟩⟩) (rawTerms := some (Proof.Events1039.exact266120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound266118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 267291)
      LeftBound267291.bound (LeftBound267291.actual selector witness) := by
  exact .transfer (LeftBound267291.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound266118.bound LeftBound267291.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound266118.bound, LeftBound267291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound266118.actual selector witness) * (LeftBound267291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267292

namespace LeftBound267387
def owner : Owner := ⟨.program ⟨257⟩, ⟨42723⟩⟩
def transferEvent : Nat := 267387
def frameStart : Nat := 267348
def rule : BoundRule := .identity (.predecessor 0 267386 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267386 .coefficient)
      LeftAuthority267384.bound (LeftAuthority267384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267384.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267384.derived selector witness)

def rawBound : CoeffClass := LeftAuthority267384.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority267384.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound267387

namespace LeftBound267404
def owner : Owner := ⟨.program ⟨257⟩, ⟨44114⟩⟩
def transferEvent : Nat := 267404
def frameStart : Nat := 267348
def rule : BoundRule := .sum [.predecessor 0 267402 .coefficient, .predecessor 1 267403 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267402 .coefficient)
      LeftBound267387.bound (LeftBound267387.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound267387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267403 .coefficient)
      LeftAuthority267400.bound (LeftAuthority267400.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority267400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267387.bound, LeftAuthority267400.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267387.bound, LeftAuthority267400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267387.actual selector witness, LeftAuthority267400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267404

namespace LeftBound267407
def owner : Owner := ⟨.program ⟨257⟩, ⟨44115⟩⟩
def transferEvent : Nat := 267407
def frameStart : Nat := 267348
def rule : BoundRule := .identity (.predecessor 0 267406 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267406 .coefficient)
      LeftBound267404.bound (LeftBound267404.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound267404.derived selector witness)

def rawBound : CoeffClass := LeftBound267404.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound267404.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound267407

namespace LeftBound267413
def owner : Owner := ⟨.program ⟨257⟩, ⟨44116⟩⟩
def transferEvent : Nat := 267413
def frameStart : Nat := 267348
def rule : BoundRule := .product (.predecessor 0 267411 .coefficient) (.predecessor 1 267412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267411 .coefficient)
      LeftAuthority267409.bound (LeftAuthority267409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267409.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267412 .coefficient)
      LeftBound267407.bound (LeftBound267407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267407.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority267409.bound LeftBound267407.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267409.bound, LeftBound267407.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority267409.actual selector witness) * (LeftBound267407.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267413

namespace LeftBound267421
def owner : Owner := ⟨.program ⟨257⟩, ⟨44117⟩⟩
def transferEvent : Nat := 267421
def frameStart : Nat := 267348
def rule : BoundRule := .sum [.predecessor 0 267419 .coefficient, .predecessor 1 267420 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267419 .coefficient)
      LeftAuthority267417.bound (LeftAuthority267417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267417.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267420 .coefficient)
      LeftBound267413.bound (LeftBound267413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority267417.bound, LeftBound267413.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267417.bound, LeftBound267413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority267417.actual selector witness, LeftBound267413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267421

namespace LeftBound267425
def owner : Owner := ⟨.program ⟨257⟩, ⟨44463⟩⟩
def transferEvent : Nat := 267425
def frameStart : Nat := 267348
def rule : BoundRule := .product (.predecessor 0 267423 .coefficient) (.predecessor 1 267424 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267423 .coefficient)
      LeftBound267421.bound (LeftBound267421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267421.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267421.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267424 .coefficient)
      LeftAuthority267398.bound (LeftAuthority267398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267398.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267398.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound267421.bound LeftAuthority267398.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267421.bound, LeftAuthority267398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound267421.actual selector witness) * (LeftAuthority267398.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267425

namespace LeftBound267436
def owner : Owner := ⟨.program ⟨257⟩, ⟨42893⟩⟩
def transferEvent : Nat := 267436
def frameStart : Nat := 267348
def rule : BoundRule := .product (.predecessor 0 267434 .coefficient) (.predecessor 1 267435 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267434 .coefficient)
      LeftAuthority267409.bound (LeftAuthority267409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267409.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267435 .coefficient)
      LeftAuthority267432.bound (LeftAuthority267432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority267409.bound LeftAuthority267432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267409.bound, LeftAuthority267432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority267409.actual selector witness) * (LeftAuthority267432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound267436

namespace LeftBound267444
def owner : Owner := ⟨.program ⟨257⟩, ⟨42894⟩⟩
def transferEvent : Nat := 267444
def frameStart : Nat := 267348
def rule : BoundRule := .sum [.predecessor 0 267442 .coefficient, .predecessor 1 267443 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267442 .coefficient)
      LeftAuthority267440.bound (LeftAuthority267440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority267440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority267440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267443 .coefficient)
      LeftBound267436.bound (LeftBound267436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267436.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267436.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority267440.bound, LeftBound267436.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority267440.bound, LeftBound267436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority267440.actual selector witness, LeftBound267436.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267444

namespace LeftBound267448
def owner : Owner := ⟨.program ⟨257⟩, ⟨44466⟩⟩
def transferEvent : Nat := 267448
def frameStart : Nat := 267348
def rule : BoundRule := .sum [.predecessor 0 267446 .coefficient, .predecessor 1 267447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267446 .coefficient)
      LeftBound267444.bound (LeftBound267444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267444.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267447 .coefficient)
      LeftBound267425.bound (LeftBound267425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267444.bound, LeftBound267425.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267444.bound, LeftBound267425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267444.actual selector witness, LeftBound267425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267448

namespace LeftBound267461
def owner : Owner := ⟨.program ⟨257⟩, ⟨44465⟩⟩
def transferEvent : Nat := 267461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 267459 .coefficient, .predecessor 1 267460 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 267459 .coefficient)
      LeftBound267290.bound (LeftBound267290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 267460 .coefficient)
      LeftBound267273.bound (LeftBound267273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1044.exact267280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound267273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound267273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267290.bound, LeftBound267273.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267290.bound, LeftBound267273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267290.actual selector witness, LeftBound267273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267461

namespace LeftBound267464
def owner : Owner := ⟨.program ⟨257⟩, ⟨44465⟩⟩
def transferEvent : Nat := 267464
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 267458 .summary, .result 267280 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267458 .summary)
      LeftBound267292.bound (LeftBound267292.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43373⟩⟩) (rawTerms := some (Proof.Events1044.exact267458RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound267292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 267280 .summary)
      LeftBound267275.bound (LeftBound267275.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44464⟩⟩) (rawTerms := some (Proof.Events1044.exact267280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound267275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound267292.bound, LeftBound267275.bound]
def bound : CoeffClass := .finite ⟨32193718473625891320532869316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound267292.bound, LeftBound267275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound267292.actual selector witness, LeftBound267275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound267464

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
