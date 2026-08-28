import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard555

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87283
def owner : Owner := ⟨.program ⟨257⟩, ⟨35614⟩⟩
def transferEvent : Nat := 87283
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87281 .coefficient) (.value (.predecessor 1 87282 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87281 .coefficient)
      LeftAuthority87279.bound (LeftAuthority87279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87282 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87279.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87279.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority87279.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87283

namespace LeftBound87287
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def transferEvent : Nat := 87287
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87285 .coefficient) (.predecessor 1 87286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87285 .coefficient)
      LeftBound75992.bound (LeftBound75992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87286 .coefficient)
      LeftBound87283.bound (LeftBound87283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87283.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75992.bound LeftBound87283.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75992.bound, LeftBound87283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75992.actual selector witness) * (LeftBound87283.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87287

namespace LeftBound87288
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def transferEvent : Nat := 87288
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩ [⟨.result 87280 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87280 .coefficient)
      LeftAuthority87279.bound (LeftAuthority87279.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35612⟩⟩) (rawTerms := some (Proof.Events340.exact87280RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87279.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87279.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority87279.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87288

namespace LeftBound87289
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def transferEvent : Nat := 87289
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 87288) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 87288)
      LeftBound87288.bound (LeftBound87288.actual selector witness) := by
  exact .transfer (LeftBound87288.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound87288.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound87288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound87288.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87289

namespace LeftBound87384
def owner : Owner := ⟨.program ⟨257⟩, ⟨34797⟩⟩
def transferEvent : Nat := 87384
def frameStart : Nat := 87345
def rule : BoundRule := .identity (.predecessor 0 87383 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87383 .coefficient)
      LeftAuthority87381.bound (LeftAuthority87381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87381.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87381.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority87381.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87384

namespace LeftBound87401
def owner : Owner := ⟨.program ⟨257⟩, ⟨36130⟩⟩
def transferEvent : Nat := 87401
def frameStart : Nat := 87345
def rule : BoundRule := .sum [.predecessor 0 87399 .coefficient, .predecessor 1 87400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87399 .coefficient)
      LeftBound87384.bound (LeftBound87384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87400 .coefficient)
      LeftAuthority87397.bound (LeftAuthority87397.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87384.bound, LeftAuthority87397.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87384.bound, LeftAuthority87397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87384.actual selector witness, LeftAuthority87397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87401

namespace LeftBound87404
def owner : Owner := ⟨.program ⟨257⟩, ⟨36131⟩⟩
def transferEvent : Nat := 87404
def frameStart : Nat := 87345
def rule : BoundRule := .identity (.predecessor 0 87403 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87403 .coefficient)
      LeftBound87401.bound (LeftBound87401.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87401.derived selector witness)

def rawBound : CoeffClass := LeftBound87401.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound87401.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87404

namespace LeftBound87410
def owner : Owner := ⟨.program ⟨257⟩, ⟨36132⟩⟩
def transferEvent : Nat := 87410
def frameStart : Nat := 87345
def rule : BoundRule := .product (.predecessor 0 87408 .coefficient) (.predecessor 1 87409 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87408 .coefficient)
      LeftAuthority87406.bound (LeftAuthority87406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87409 .coefficient)
      LeftBound87404.bound (LeftBound87404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority87406.bound LeftBound87404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87406.bound, LeftBound87404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority87406.actual selector witness) * (LeftBound87404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87410

namespace LeftBound87418
def owner : Owner := ⟨.program ⟨257⟩, ⟨36133⟩⟩
def transferEvent : Nat := 87418
def frameStart : Nat := 87345
def rule : BoundRule := .sum [.predecessor 0 87416 .coefficient, .predecessor 1 87417 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87416 .coefficient)
      LeftAuthority87414.bound (LeftAuthority87414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87414.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87414.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87417 .coefficient)
      LeftBound87410.bound (LeftBound87410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87414.bound, LeftBound87410.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87414.bound, LeftBound87410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority87414.actual selector witness, LeftBound87410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87418

namespace LeftBound87422
def owner : Owner := ⟨.program ⟨257⟩, ⟨36774⟩⟩
def transferEvent : Nat := 87422
def frameStart : Nat := 87345
def rule : BoundRule := .product (.predecessor 0 87420 .coefficient) (.predecessor 1 87421 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87420 .coefficient)
      LeftBound87418.bound (LeftBound87418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87421 .coefficient)
      LeftAuthority87395.bound (LeftAuthority87395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87395.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound87418.bound LeftAuthority87395.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87418.bound, LeftAuthority87395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound87418.actual selector witness) * (LeftAuthority87395.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87422

namespace LeftBound87433
def owner : Owner := ⟨.program ⟨257⟩, ⟨35039⟩⟩
def transferEvent : Nat := 87433
def frameStart : Nat := 87345
def rule : BoundRule := .product (.predecessor 0 87431 .coefficient) (.predecessor 1 87432 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87431 .coefficient)
      LeftAuthority87406.bound (LeftAuthority87406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87432 .coefficient)
      LeftAuthority87429.bound (LeftAuthority87429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87429.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87406.bound LeftAuthority87429.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87406.bound, LeftAuthority87429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority87406.actual selector witness) * (LeftAuthority87429.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87433

namespace LeftBound87441
def owner : Owner := ⟨.program ⟨257⟩, ⟨35040⟩⟩
def transferEvent : Nat := 87441
def frameStart : Nat := 87345
def rule : BoundRule := .sum [.predecessor 0 87439 .coefficient, .predecessor 1 87440 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87439 .coefficient)
      LeftAuthority87437.bound (LeftAuthority87437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87440 .coefficient)
      LeftBound87433.bound (LeftBound87433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87433.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87433.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87437.bound, LeftBound87433.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87437.bound, LeftBound87433.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority87437.actual selector witness, LeftBound87433.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87441

namespace LeftBound87445
def owner : Owner := ⟨.program ⟨257⟩, ⟨36778⟩⟩
def transferEvent : Nat := 87445
def frameStart : Nat := 87345
def rule : BoundRule := .sum [.predecessor 0 87443 .coefficient, .predecessor 1 87444 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87443 .coefficient)
      LeftBound87441.bound (LeftBound87441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87444 .coefficient)
      LeftBound87422.bound (LeftBound87422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87422.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87441.bound, LeftBound87422.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87441.bound, LeftBound87422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87441.actual selector witness, LeftBound87422.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87445

namespace LeftBound87458
def owner : Owner := ⟨.program ⟨257⟩, ⟨36776⟩⟩
def transferEvent : Nat := 87458
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87456 .coefficient, .predecessor 1 87457 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87456 .coefficient)
      LeftBound87287.bound (LeftBound87287.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87287.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87287.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87457 .coefficient)
      LeftBound87270.bound (LeftBound87270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87270.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87287.bound, LeftBound87270.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87287.bound, LeftBound87270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87287.actual selector witness, LeftBound87270.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87458

namespace LeftBound87461
def owner : Owner := ⟨.program ⟨257⟩, ⟨36776⟩⟩
def transferEvent : Nat := 87461
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87455 .summary, .result 87277 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87455 .summary)
      LeftBound87289.bound (LeftBound87289.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35615⟩⟩) (rawTerms := some (Proof.Events341.exact87455RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87277 .summary)
      LeftBound87272.bound (LeftBound87272.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36775⟩⟩) (rawTerms := some (Proof.Events340.exact87277RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87272.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87289.bound, LeftBound87272.bound]
def bound : CoeffClass := .finite ⟨32192539770951767057087530795008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87289.bound, LeftBound87272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87289.actual selector witness, LeftBound87272.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87461

namespace LeftBound87465
def owner : Owner := ⟨.program ⟨257⟩, ⟨36777⟩⟩
def transferEvent : Nat := 87465
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87463 .coefficient) (.predecessor 1 87464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87463 .coefficient)
      LeftBound87458.bound (LeftBound87458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events341.exact87462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87464 .coefficient)
      LeftBound15641.bound (LeftBound15641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound87458.bound LeftBound15641.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87458.bound, LeftBound15641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound87458.actual selector witness) * (LeftBound15641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87465

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
