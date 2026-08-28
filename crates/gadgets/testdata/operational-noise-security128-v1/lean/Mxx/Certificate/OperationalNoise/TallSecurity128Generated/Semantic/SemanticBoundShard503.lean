import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard475
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard478
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard479
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard502

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79094
def owner : Owner := ⟨.program ⟨257⟩, ⟨29959⟩⟩
def transferEvent : Nat := 79094
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨29956⟩⟩]⟩ [⟨.result 79086 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79086 .coefficient)
      LeftAuthority79085.bound (LeftAuthority79085.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29956⟩⟩) (rawTerms := some (Proof.Events308.exact79086RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79085.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79085.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority79085.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79094

namespace LeftBound79095
def owner : Owner := ⟨.program ⟨257⟩, ⟨29959⟩⟩
def transferEvent : Nat := 79095
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 75995 .summary) (.transfer 79094) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 75995 .summary)
      LeftBound75993.bound (LeftBound75993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨10368⟩⟩) (rawTerms := some (Proof.Events296.exact75995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 79094)
      LeftBound79094.bound (LeftBound79094.actual selector witness) := by
  exact .transfer (LeftBound79094.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound75993.bound LeftBound79094.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75993.bound, LeftBound79094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound75993.actual selector witness) * (LeftBound79094.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79095

namespace LeftBound79190
def owner : Owner := ⟨.program ⟨257⟩, ⟨29137⟩⟩
def transferEvent : Nat := 79190
def frameStart : Nat := 79151
def rule : BoundRule := .identity (.predecessor 0 79189 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79189 .coefficient)
      LeftAuthority79187.bound (LeftAuthority79187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79187.derived selector witness)

def rawBound : CoeffClass := LeftAuthority79187.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority79187.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79190

namespace LeftBound79207
def owner : Owner := ⟨.program ⟨257⟩, ⟨30470⟩⟩
def transferEvent : Nat := 79207
def frameStart : Nat := 79151
def rule : BoundRule := .sum [.predecessor 0 79205 .coefficient, .predecessor 1 79206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79205 .coefficient)
      LeftBound79190.bound (LeftBound79190.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79206 .coefficient)
      LeftAuthority79203.bound (LeftAuthority79203.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority79203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79190.bound, LeftAuthority79203.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79190.bound, LeftAuthority79203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79190.actual selector witness, LeftAuthority79203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79207

namespace LeftBound79210
def owner : Owner := ⟨.program ⟨257⟩, ⟨30471⟩⟩
def transferEvent : Nat := 79210
def frameStart : Nat := 79151
def rule : BoundRule := .identity (.predecessor 0 79209 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79209 .coefficient)
      LeftBound79207.bound (LeftBound79207.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79207.derived selector witness)

def rawBound : CoeffClass := LeftBound79207.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound79207.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79210

namespace LeftBound79216
def owner : Owner := ⟨.program ⟨257⟩, ⟨30472⟩⟩
def transferEvent : Nat := 79216
def frameStart : Nat := 79151
def rule : BoundRule := .product (.predecessor 0 79214 .coefficient) (.predecessor 1 79215 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79214 .coefficient)
      LeftAuthority79212.bound (LeftAuthority79212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79215 .coefficient)
      LeftBound79210.bound (LeftBound79210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79210.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority79212.bound LeftBound79210.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79212.bound, LeftBound79210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority79212.actual selector witness) * (LeftBound79210.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79216

namespace LeftBound79224
def owner : Owner := ⟨.program ⟨257⟩, ⟨30473⟩⟩
def transferEvent : Nat := 79224
def frameStart : Nat := 79151
def rule : BoundRule := .sum [.predecessor 0 79222 .coefficient, .predecessor 1 79223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79222 .coefficient)
      LeftAuthority79220.bound (LeftAuthority79220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79223 .coefficient)
      LeftBound79216.bound (LeftBound79216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79216.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79220.bound, LeftBound79216.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79220.bound, LeftBound79216.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority79220.actual selector witness, LeftBound79216.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79224

namespace LeftBound79228
def owner : Owner := ⟨.program ⟨257⟩, ⟨31120⟩⟩
def transferEvent : Nat := 79228
def frameStart : Nat := 79151
def rule : BoundRule := .product (.predecessor 0 79226 .coefficient) (.predecessor 1 79227 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79226 .coefficient)
      LeftBound79224.bound (LeftBound79224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79227 .coefficient)
      LeftAuthority79201.bound (LeftAuthority79201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79201.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79201.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound79224.bound LeftAuthority79201.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79224.bound, LeftAuthority79201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound79224.actual selector witness) * (LeftAuthority79201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79228

namespace LeftBound79239
def owner : Owner := ⟨.program ⟨257⟩, ⟨29378⟩⟩
def transferEvent : Nat := 79239
def frameStart : Nat := 79151
def rule : BoundRule := .product (.predecessor 0 79237 .coefficient) (.predecessor 1 79238 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79237 .coefficient)
      LeftAuthority79212.bound (LeftAuthority79212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79212.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79238 .coefficient)
      LeftAuthority79235.bound (LeftAuthority79235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority79212.bound LeftAuthority79235.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79212.bound, LeftAuthority79235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority79212.actual selector witness) * (LeftAuthority79235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79239

namespace LeftBound79247
def owner : Owner := ⟨.program ⟨257⟩, ⟨29379⟩⟩
def transferEvent : Nat := 79247
def frameStart : Nat := 79151
def rule : BoundRule := .sum [.predecessor 0 79245 .coefficient, .predecessor 1 79246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79245 .coefficient)
      LeftAuthority79243.bound (LeftAuthority79243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79246 .coefficient)
      LeftBound79239.bound (LeftBound79239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79239.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority79243.bound, LeftBound79239.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79243.bound, LeftBound79239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority79243.actual selector witness, LeftBound79239.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79247

namespace LeftBound79251
def owner : Owner := ⟨.program ⟨257⟩, ⟨31123⟩⟩
def transferEvent : Nat := 79251
def frameStart : Nat := 79151
def rule : BoundRule := .sum [.predecessor 0 79249 .coefficient, .predecessor 1 79250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79249 .coefficient)
      LeftBound79247.bound (LeftBound79247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79250 .coefficient)
      LeftBound79228.bound (LeftBound79228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79247.bound, LeftBound79228.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79247.bound, LeftBound79228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79247.actual selector witness, LeftBound79228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79251

namespace LeftBound79264
def owner : Owner := ⟨.program ⟨257⟩, ⟨31122⟩⟩
def transferEvent : Nat := 79264
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79262 .coefficient, .predecessor 1 79263 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79262 .coefficient)
      LeftBound79093.bound (LeftBound79093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79263 .coefficient)
      LeftBound79076.bound (LeftBound79076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events308.exact79083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79076.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79093.bound, LeftBound79076.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79093.bound, LeftBound79076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79093.actual selector witness, LeftBound79076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79264

namespace LeftBound79267
def owner : Owner := ⟨.program ⟨257⟩, ⟨31122⟩⟩
def transferEvent : Nat := 79267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79261 .summary, .result 79083 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79261 .summary)
      LeftBound79095.bound (LeftBound79095.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨29959⟩⟩) (rawTerms := some (Proof.Events309.exact79261RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79083 .summary)
      LeftBound79078.bound (LeftBound79078.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31121⟩⟩) (rawTerms := some (Proof.Events308.exact79083RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79095.bound, LeftBound79078.bound]
def bound : CoeffClass := .finite ⟨32192146870060392302605751287808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79095.bound, LeftBound79078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79095.actual selector witness, LeftBound79078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79267

namespace LeftBound79291
def owner : Owner := ⟨.program ⟨257⟩, ⟨26241⟩⟩
def transferEvent : Nat := 79291
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 79289 .coefficient) (.predecessor 1 79290 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79289 .coefficient)
      LeftAuthority3246.bound (LeftAuthority3246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79290 .coefficient)
      LeftBound75901.bound (LeftBound75901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority3246.bound LeftBound75901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3246.bound, LeftBound75901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority3246.actual selector witness) * (LeftBound75901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79291

namespace LeftBound79296
def owner : Owner := ⟨.program ⟨257⟩, ⟨10336⟩⟩
def transferEvent : Nat := 79296
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79294 .coefficient) (.predecessor 1 79295 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79294 .coefficient)
      LeftBound75772.bound (LeftBound75772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events295.exact75773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79295 .coefficient)
      LeftBound20586.bound (LeftBound20586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound75772.bound LeftBound20586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75772.bound, LeftBound20586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound75772.actual selector witness) * (LeftBound20586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79296

namespace LeftBound79301
def owner : Owner := ⟨.program ⟨257⟩, ⟨26242⟩⟩
def transferEvent : Nat := 79301
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79299 .coefficient, .predecessor 1 79300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79299 .coefficient)
      LeftBound79296.bound (LeftBound79296.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79296.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79296.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79300 .coefficient)
      LeftBound79291.bound (LeftBound79291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events309.exact79293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79296.bound, LeftBound79291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79296.bound, LeftBound79291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79296.actual selector witness, LeftBound79291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79301

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
