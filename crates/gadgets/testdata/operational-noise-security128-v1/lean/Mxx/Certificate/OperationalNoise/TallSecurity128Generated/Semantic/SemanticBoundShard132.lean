import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard131

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25290
def owner : Owner := ⟨.program ⟨257⟩, ⟨19950⟩⟩
def transferEvent : Nat := 25290
def frameStart : Nat := 25240
def rule : BoundRule := .sum [.predecessor 0 25288 .coefficient, .predecessor 1 25289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25288 .coefficient)
      LeftBound25273.bound (LeftBound25273.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25289 .coefficient)
      LeftAuthority25286.bound (LeftAuthority25286.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25273.bound, LeftAuthority25286.bound]
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25273.bound, LeftAuthority25286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound25273.actual selector witness, LeftAuthority25286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25290

namespace LeftBound25293
def owner : Owner := ⟨.program ⟨257⟩, ⟨19951⟩⟩
def transferEvent : Nat := 25293
def frameStart : Nat := 25240
def rule : BoundRule := .identity (.predecessor 0 25292 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25292 .coefficient)
      LeftBound25290.bound (LeftBound25290.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25290.derived selector witness)

def rawBound : CoeffClass := LeftBound25290.bound
def bound : CoeffClass := .finite ⟨9, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound25290.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25293

namespace LeftBound25299
def owner : Owner := ⟨.program ⟨257⟩, ⟨19952⟩⟩
def transferEvent : Nat := 25299
def frameStart : Nat := 25240
def rule : BoundRule := .product (.predecessor 0 25297 .coefficient) (.predecessor 1 25298 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25297 .coefficient)
      LeftAuthority25295.bound (LeftAuthority25295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25298 .coefficient)
      LeftBound25293.bound (LeftBound25293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority25295.bound LeftBound25293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25295.bound, LeftBound25293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority25295.actual selector witness) * (LeftBound25293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25299

namespace LeftBound25315
def owner : Owner := ⟨.program ⟨257⟩, ⟨9572⟩⟩
def transferEvent : Nat := 25315
def frameStart : Nat := 25240
def rule : BoundRule := .scale (.predecessor 0 25313 .coefficient) (.value (.predecessor 1 25314 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25313 .coefficient)
      LeftAuthority25311.bound (LeftAuthority25311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25311.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25314 .coefficient)
      LeftAuthority25302.bound (LeftAuthority25302.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority25302.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority25311.bound LeftAuthority25302.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25311.bound, LeftAuthority25302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority25311.actual selector witness) * (LeftAuthority25302.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25315

namespace LeftBound25318
def owner : Owner := ⟨.program ⟨257⟩, ⟨7277⟩⟩
def transferEvent : Nat := 25318
def frameStart : Nat := 25240
def rule : BoundRule := .identity (.predecessor 0 25317 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25317 .coefficient)
      LeftAuthority25305.bound (LeftAuthority25305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25305.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25305.derived selector witness)

def rawBound : CoeffClass := LeftAuthority25305.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority25305.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25318

namespace LeftBound25322
def owner : Owner := ⟨.program ⟨257⟩, ⟨9573⟩⟩
def transferEvent : Nat := 25322
def frameStart : Nat := 25240
def rule : BoundRule := .product (.predecessor 0 25320 .coefficient) (.predecessor 1 25321 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25320 .coefficient)
      LeftBound25318.bound (LeftBound25318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25321 .coefficient)
      LeftBound25315.bound (LeftBound25315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25318.bound LeftBound25315.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25318.bound, LeftBound25315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25318.actual selector witness) * (LeftBound25315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25322

namespace LeftBound25327
def owner : Owner := ⟨.program ⟨257⟩, ⟨19953⟩⟩
def transferEvent : Nat := 25327
def frameStart : Nat := 25240
def rule : BoundRule := .sum [.predecessor 0 25325 .coefficient, .predecessor 1 25326 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25325 .coefficient)
      LeftBound25322.bound (LeftBound25322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25326 .coefficient)
      LeftBound25299.bound (LeftBound25299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25299.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25322.bound, LeftBound25299.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25322.bound, LeftBound25299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound25322.actual selector witness, LeftBound25299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25327

namespace LeftBound25331
def owner : Owner := ⟨.program ⟨257⟩, ⟨20126⟩⟩
def transferEvent : Nat := 25331
def frameStart : Nat := 25240
def rule : BoundRule := .product (.predecessor 0 25329 .coefficient) (.predecessor 1 25330 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25329 .coefficient)
      LeftBound25327.bound (LeftBound25327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25330 .coefficient)
      LeftAuthority25284.bound (LeftAuthority25284.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25284.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25284.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25327.bound LeftAuthority25284.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25327.bound, LeftAuthority25284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25327.actual selector witness) * (LeftAuthority25284.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25331

namespace LeftBound25342
def owner : Owner := ⟨.program ⟨257⟩, ⟨18520⟩⟩
def transferEvent : Nat := 25342
def frameStart : Nat := 25240
def rule : BoundRule := .product (.predecessor 0 25340 .coefficient) (.predecessor 1 25341 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25340 .coefficient)
      LeftAuthority25295.bound (LeftAuthority25295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25296RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25295.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25341 .coefficient)
      LeftAuthority25338.bound (LeftAuthority25338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25338.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25295.bound LeftAuthority25338.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25295.bound, LeftAuthority25338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority25295.actual selector witness) * (LeftAuthority25338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25342

namespace LeftBound25350
def owner : Owner := ⟨.program ⟨257⟩, ⟨18521⟩⟩
def transferEvent : Nat := 25350
def frameStart : Nat := 25240
def rule : BoundRule := .sum [.predecessor 0 25348 .coefficient, .predecessor 1 25349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25348 .coefficient)
      LeftAuthority25346.bound (LeftAuthority25346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25346.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25349 .coefficient)
      LeftBound25342.bound (LeftBound25342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25342.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25346.bound, LeftBound25342.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25346.bound, LeftBound25342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority25346.actual selector witness, LeftBound25342.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25350

namespace LeftBound25354
def owner : Owner := ⟨.program ⟨257⟩, ⟨20127⟩⟩
def transferEvent : Nat := 25354
def frameStart : Nat := 25240
def rule : BoundRule := .sum [.predecessor 0 25352 .coefficient, .predecessor 1 25353 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25352 .coefficient)
      LeftBound25350.bound (LeftBound25350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25353 .coefficient)
      LeftBound25331.bound (LeftBound25331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25350.bound, LeftBound25331.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25350.bound, LeftBound25331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound25350.actual selector witness, LeftBound25331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25354

namespace LeftBound25367
def owner : Owner := ⟨.program ⟨257⟩, ⟨20125⟩⟩
def transferEvent : Nat := 25367
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25365 .coefficient, .predecessor 1 25366 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25365 .coefficient)
      LeftBound25188.bound (LeftBound25188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25366 .coefficient)
      LeftBound25171.bound (LeftBound25171.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25171.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25188.bound, LeftBound25171.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25188.bound, LeftBound25171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound25188.actual selector witness, LeftBound25171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25367

namespace LeftBound25370
def owner : Owner := ⟨.program ⟨257⟩, ⟨20125⟩⟩
def transferEvent : Nat := 25370
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25364 .summary, .result 25178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25364 .summary)
      LeftBound25190.bound (LeftBound25190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19065⟩⟩) (rawTerms := some (Proof.Events099.exact25364RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25178 .summary)
      LeftBound25173.bound (LeftBound25173.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20124⟩⟩) (rawTerms := some (Proof.Events098.exact25178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25190.bound, LeftBound25173.bound]
def bound : CoeffClass := .finite ⟨2997825428629885288448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25190.bound, LeftBound25173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound25190.actual selector witness, LeftBound25173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25370

namespace LeftBound25374
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def transferEvent : Nat := 25374
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25372 .coefficient) (.predecessor 1 25373 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 25372 .coefficient)
      LeftBound25367.bound (LeftBound25367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 25373 .coefficient)
      LeftAuthority25074.bound (LeftAuthority25074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25367.bound LeftAuthority25074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25367.bound, LeftAuthority25074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25367.actual selector witness) * (LeftAuthority25074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25374

namespace LeftBound25375
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def transferEvent : Nat := 25375
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20382⟩⟩]⟩ [⟨.result 25075 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25075 .coefficient)
      LeftAuthority25074.bound (LeftAuthority25074.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20382⟩⟩) (rawTerms := some (Proof.Events097.exact25075RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25074.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority25074.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority25074.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25375

namespace LeftBound25376
def owner : Owner := ⟨.program ⟨257⟩, ⟨20384⟩⟩
def transferEvent : Nat := 25376
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 25371 .summary) (.transfer 25375) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25371 .summary)
      LeftBound25370.bound (LeftBound25370.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20125⟩⟩) (rawTerms := some (Proof.Events099.exact25371RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25370.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 25375)
      LeftBound25375.bound (LeftBound25375.actual selector witness) := by
  exact .transfer (LeftBound25375.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound25370.bound LeftBound25375.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25370.bound, LeftBound25375.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound25370.actual selector witness) * (LeftBound25375.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25376

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
