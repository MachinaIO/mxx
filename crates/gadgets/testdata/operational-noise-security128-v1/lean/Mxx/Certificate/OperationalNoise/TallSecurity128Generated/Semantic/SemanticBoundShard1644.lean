import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1643

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound243327
def owner : Owner := ⟨.program ⟨257⟩, ⟨52892⟩⟩
def transferEvent : Nat := 243327
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 243322 .summary) (.transfer 243326) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243322 .summary)
      LeftBound243321.bound (LeftBound243321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52499⟩⟩) (rawTerms := some (Proof.Events950.exact243322RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 243326)
      LeftBound243326.bound (LeftBound243326.actual selector witness) := by
  exact .transfer (LeftBound243326.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound243321.bound LeftBound243326.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243321.bound, LeftBound243326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound243321.actual selector witness) * (LeftBound243326.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243327

namespace LeftBound243338
def owner : Owner := ⟨.program ⟨257⟩, ⟨51718⟩⟩
def transferEvent : Nat := 243338
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 243336 .coefficient) (.value (.predecessor 1 243337 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243336 .coefficient)
      LeftAuthority243334.bound (LeftAuthority243334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243337 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority243334.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243334.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority243334.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound243338

namespace LeftBound243342
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def transferEvent : Nat := 243342
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 243340 .coefficient) (.predecessor 1 243341 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243340 .coefficient)
      LeftBound236867.bound (LeftBound236867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243341 .coefficient)
      LeftBound243338.bound (LeftBound243338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236867.bound LeftBound243338.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236867.bound, LeftBound243338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236867.actual selector witness) * (LeftBound243338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243342

namespace LeftBound243343
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def transferEvent : Nat := 243343
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51716⟩⟩]⟩ [⟨.result 243335 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243335 .coefficient)
      LeftAuthority243334.bound (LeftAuthority243334.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51716⟩⟩) (rawTerms := some (Proof.Events950.exact243335RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243334.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority243334.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority243334.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound243343

namespace LeftBound243344
def owner : Owner := ⟨.program ⟨257⟩, ⟨51719⟩⟩
def transferEvent : Nat := 243344
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 243343) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 243343)
      LeftBound243343.bound (LeftBound243343.actual selector witness) := by
  exact .transfer (LeftBound243343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound243343.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound243343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound243343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243344

namespace LeftBound243439
def owner : Owner := ⟨.program ⟨257⟩, ⟨50873⟩⟩
def transferEvent : Nat := 243439
def frameStart : Nat := 243400
def rule : BoundRule := .identity (.predecessor 0 243438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243438 .coefficient)
      LeftAuthority243436.bound (LeftAuthority243436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243436.derived selector witness)

def rawBound : CoeffClass := LeftAuthority243436.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority243436.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound243439

namespace LeftBound243456
def owner : Owner := ⟨.program ⟨257⟩, ⟨52358⟩⟩
def transferEvent : Nat := 243456
def frameStart : Nat := 243400
def rule : BoundRule := .sum [.predecessor 0 243454 .coefficient, .predecessor 1 243455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243454 .coefficient)
      LeftBound243439.bound (LeftBound243439.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound243439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243455 .coefficient)
      LeftAuthority243452.bound (LeftAuthority243452.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority243452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243439.bound, LeftAuthority243452.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243439.bound, LeftAuthority243452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243439.actual selector witness, LeftAuthority243452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243456

namespace LeftBound243459
def owner : Owner := ⟨.program ⟨257⟩, ⟨52359⟩⟩
def transferEvent : Nat := 243459
def frameStart : Nat := 243400
def rule : BoundRule := .identity (.predecessor 0 243458 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243458 .coefficient)
      LeftBound243456.bound (LeftBound243456.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound243456.derived selector witness)

def rawBound : CoeffClass := LeftBound243456.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound243456.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound243459

namespace LeftBound243465
def owner : Owner := ⟨.program ⟨257⟩, ⟨52360⟩⟩
def transferEvent : Nat := 243465
def frameStart : Nat := 243400
def rule : BoundRule := .product (.predecessor 0 243463 .coefficient) (.predecessor 1 243464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243463 .coefficient)
      LeftAuthority243461.bound (LeftAuthority243461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243464 .coefficient)
      LeftBound243459.bound (LeftBound243459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority243461.bound LeftBound243459.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243461.bound, LeftBound243459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority243461.actual selector witness) * (LeftBound243459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243465

namespace LeftBound243473
def owner : Owner := ⟨.program ⟨257⟩, ⟨52361⟩⟩
def transferEvent : Nat := 243473
def frameStart : Nat := 243400
def rule : BoundRule := .sum [.predecessor 0 243471 .coefficient, .predecessor 1 243472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243471 .coefficient)
      LeftAuthority243469.bound (LeftAuthority243469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243469.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243472 .coefficient)
      LeftBound243465.bound (LeftBound243465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority243469.bound, LeftBound243465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243469.bound, LeftBound243465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority243469.actual selector witness, LeftBound243465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243473

namespace LeftBound243477
def owner : Owner := ⟨.program ⟨257⟩, ⟨52891⟩⟩
def transferEvent : Nat := 243477
def frameStart : Nat := 243400
def rule : BoundRule := .product (.predecessor 0 243475 .coefficient) (.predecessor 1 243476 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243475 .coefficient)
      LeftBound243473.bound (LeftBound243473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243476 .coefficient)
      LeftAuthority243450.bound (LeftAuthority243450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound243473.bound LeftAuthority243450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243473.bound, LeftAuthority243450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound243473.actual selector witness) * (LeftAuthority243450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243477

namespace LeftBound243488
def owner : Owner := ⟨.program ⟨257⟩, ⟨51125⟩⟩
def transferEvent : Nat := 243488
def frameStart : Nat := 243400
def rule : BoundRule := .product (.predecessor 0 243486 .coefficient) (.predecessor 1 243487 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243486 .coefficient)
      LeftAuthority243461.bound (LeftAuthority243461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243487 .coefficient)
      LeftAuthority243484.bound (LeftAuthority243484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243484.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority243461.bound LeftAuthority243484.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243461.bound, LeftAuthority243484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority243461.actual selector witness) * (LeftAuthority243484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound243488

namespace LeftBound243496
def owner : Owner := ⟨.program ⟨257⟩, ⟨51126⟩⟩
def transferEvent : Nat := 243496
def frameStart : Nat := 243400
def rule : BoundRule := .sum [.predecessor 0 243494 .coefficient, .predecessor 1 243495 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243494 .coefficient)
      LeftAuthority243492.bound (LeftAuthority243492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority243492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority243492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243495 .coefficient)
      LeftBound243488.bound (LeftBound243488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority243492.bound, LeftBound243488.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority243492.bound, LeftBound243488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority243492.actual selector witness, LeftBound243488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243496

namespace LeftBound243500
def owner : Owner := ⟨.program ⟨257⟩, ⟨52895⟩⟩
def transferEvent : Nat := 243500
def frameStart : Nat := 243400
def rule : BoundRule := .sum [.predecessor 0 243498 .coefficient, .predecessor 1 243499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243498 .coefficient)
      LeftBound243496.bound (LeftBound243496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243499 .coefficient)
      LeftBound243477.bound (LeftBound243477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243496.bound, LeftBound243477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243496.bound, LeftBound243477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243496.actual selector witness, LeftBound243477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243500

namespace LeftBound243513
def owner : Owner := ⟨.program ⟨257⟩, ⟨52893⟩⟩
def transferEvent : Nat := 243513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 243511 .coefficient, .predecessor 1 243512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 243511 .coefficient)
      LeftBound243342.bound (LeftBound243342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events951.exact243510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 243512 .coefficient)
      LeftBound243325.bound (LeftBound243325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events950.exact243332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound243325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound243325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243342.bound, LeftBound243325.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243342.bound, LeftBound243325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243342.actual selector witness, LeftBound243325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243513

namespace LeftBound243516
def owner : Owner := ⟨.program ⟨257⟩, ⟨52893⟩⟩
def transferEvent : Nat := 243516
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 243510 .summary, .result 243332 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243510 .summary)
      LeftBound243344.bound (LeftBound243344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51719⟩⟩) (rawTerms := some (Proof.Events951.exact243510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 243332 .summary)
      LeftBound243327.bound (LeftBound243327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52892⟩⟩) (rawTerms := some (Proof.Events950.exact243332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound243327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound243344.bound, LeftBound243327.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound243344.bound, LeftBound243327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound243344.actual selector witness, LeftBound243327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound243516

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
