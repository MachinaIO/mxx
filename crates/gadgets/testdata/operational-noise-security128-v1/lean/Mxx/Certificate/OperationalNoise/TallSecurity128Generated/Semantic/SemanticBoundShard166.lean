import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard165

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31328
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def transferEvent : Nat := 31328
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31326 .coefficient) (.predecessor 1 31327 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31326 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31327 .coefficient)
      LeftBound31324.bound (LeftBound31324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound31324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound31324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound31324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31328

namespace LeftBound31329
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def transferEvent : Nat := 31329
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨16418⟩⟩]⟩ [⟨.result 31321 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31321 .coefficient)
      LeftAuthority31320.bound (LeftAuthority31320.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨16418⟩⟩) (rawTerms := some (Proof.Events122.exact31321RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31320.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31320.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority31320.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority31320.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31329

namespace LeftBound31330
def owner : Owner := ⟨.program ⟨257⟩, ⟨16421⟩⟩
def transferEvent : Nat := 31330
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 31329) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 31329)
      LeftBound31329.bound (LeftBound31329.actual selector witness) := by
  exact .transfer (LeftBound31329.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound31329.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound31329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound31329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31330

namespace LeftBound31425
def owner : Owner := ⟨.program ⟨257⟩, ⟨15719⟩⟩
def transferEvent : Nat := 31425
def frameStart : Nat := 31386
def rule : BoundRule := .identity (.predecessor 0 31424 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31424 .coefficient)
      LeftAuthority31422.bound (LeftAuthority31422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31422.derived selector witness)

def rawBound : CoeffClass := LeftAuthority31422.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority31422.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31425

namespace LeftBound31442
def owner : Owner := ⟨.program ⟨257⟩, ⟨17170⟩⟩
def transferEvent : Nat := 31442
def frameStart : Nat := 31386
def rule : BoundRule := .sum [.predecessor 0 31440 .coefficient, .predecessor 1 31441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31440 .coefficient)
      LeftBound31425.bound (LeftBound31425.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31441 .coefficient)
      LeftAuthority31438.bound (LeftAuthority31438.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority31438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31425.bound, LeftAuthority31438.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31425.bound, LeftAuthority31438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31425.actual selector witness, LeftAuthority31438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31442

namespace LeftBound31445
def owner : Owner := ⟨.program ⟨257⟩, ⟨17171⟩⟩
def transferEvent : Nat := 31445
def frameStart : Nat := 31386
def rule : BoundRule := .identity (.predecessor 0 31444 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31444 .coefficient)
      LeftBound31442.bound (LeftBound31442.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound31442.derived selector witness)

def rawBound : CoeffClass := LeftBound31442.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound31442.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound31445

namespace LeftBound31451
def owner : Owner := ⟨.program ⟨257⟩, ⟨17172⟩⟩
def transferEvent : Nat := 31451
def frameStart : Nat := 31386
def rule : BoundRule := .product (.predecessor 0 31449 .coefficient) (.predecessor 1 31450 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31449 .coefficient)
      LeftAuthority31447.bound (LeftAuthority31447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31450 .coefficient)
      LeftBound31445.bound (LeftBound31445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31445.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority31447.bound LeftBound31445.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31447.bound, LeftBound31445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority31447.actual selector witness) * (LeftBound31445.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31451

namespace LeftBound31459
def owner : Owner := ⟨.program ⟨257⟩, ⟨17173⟩⟩
def transferEvent : Nat := 31459
def frameStart : Nat := 31386
def rule : BoundRule := .sum [.predecessor 0 31457 .coefficient, .predecessor 1 31458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31457 .coefficient)
      LeftAuthority31455.bound (LeftAuthority31455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31458 .coefficient)
      LeftBound31451.bound (LeftBound31451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31451.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31455.bound, LeftBound31451.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31455.bound, LeftBound31451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority31455.actual selector witness, LeftBound31451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31459

namespace LeftBound31463
def owner : Owner := ⟨.program ⟨257⟩, ⟨17511⟩⟩
def transferEvent : Nat := 31463
def frameStart : Nat := 31386
def rule : BoundRule := .product (.predecessor 0 31461 .coefficient) (.predecessor 1 31462 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31461 .coefficient)
      LeftBound31459.bound (LeftBound31459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31462 .coefficient)
      LeftAuthority31436.bound (LeftAuthority31436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31436.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound31459.bound LeftAuthority31436.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31459.bound, LeftAuthority31436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound31459.actual selector witness) * (LeftAuthority31436.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31463

namespace LeftBound31474
def owner : Owner := ⟨.program ⟨257⟩, ⟨15893⟩⟩
def transferEvent : Nat := 31474
def frameStart : Nat := 31386
def rule : BoundRule := .product (.predecessor 0 31472 .coefficient) (.predecessor 1 31473 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31472 .coefficient)
      LeftAuthority31447.bound (LeftAuthority31447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31447.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31473 .coefficient)
      LeftAuthority31470.bound (LeftAuthority31470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31470.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority31447.bound LeftAuthority31470.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31447.bound, LeftAuthority31470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority31447.actual selector witness) * (LeftAuthority31470.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31474

namespace LeftBound31482
def owner : Owner := ⟨.program ⟨257⟩, ⟨15894⟩⟩
def transferEvent : Nat := 31482
def frameStart : Nat := 31386
def rule : BoundRule := .sum [.predecessor 0 31480 .coefficient, .predecessor 1 31481 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31480 .coefficient)
      LeftAuthority31478.bound (LeftAuthority31478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority31478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority31478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31481 .coefficient)
      LeftBound31474.bound (LeftBound31474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31474.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority31478.bound, LeftBound31474.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority31478.bound, LeftBound31474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority31478.actual selector witness, LeftBound31474.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31482

namespace LeftBound31486
def owner : Owner := ⟨.program ⟨257⟩, ⟨17516⟩⟩
def transferEvent : Nat := 31486
def frameStart : Nat := 31386
def rule : BoundRule := .sum [.predecessor 0 31484 .coefficient, .predecessor 1 31485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31484 .coefficient)
      LeftBound31482.bound (LeftBound31482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31482.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31485 .coefficient)
      LeftBound31463.bound (LeftBound31463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31482.bound, LeftBound31463.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31482.bound, LeftBound31463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31482.actual selector witness, LeftBound31463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31486

namespace LeftBound31499
def owner : Owner := ⟨.program ⟨257⟩, ⟨17513⟩⟩
def transferEvent : Nat := 31499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31497 .coefficient, .predecessor 1 31498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31497 .coefficient)
      LeftBound31328.bound (LeftBound31328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31498 .coefficient)
      LeftBound31311.bound (LeftBound31311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events122.exact31318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31328.bound, LeftBound31311.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31328.bound, LeftBound31311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31328.actual selector witness, LeftBound31311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31499

namespace LeftBound31502
def owner : Owner := ⟨.program ⟨257⟩, ⟨17513⟩⟩
def transferEvent : Nat := 31502
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31496 .summary, .result 31318 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31496 .summary)
      LeftBound31330.bound (LeftBound31330.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16421⟩⟩) (rawTerms := some (Proof.Events123.exact31496RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31330.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31318 .summary)
      LeftBound31313.bound (LeftBound31313.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17512⟩⟩) (rawTerms := some (Proof.Events122.exact31318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31330.bound, LeftBound31313.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31330.bound, LeftBound31313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31330.actual selector witness, LeftBound31313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31502

namespace LeftBound31506
def owner : Owner := ⟨.program ⟨257⟩, ⟨17514⟩⟩
def transferEvent : Nat := 31506
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31504 .coefficient) (.predecessor 1 31505 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31504 .coefficient)
      LeftBound31499.bound (LeftBound31499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31505 .coefficient)
      LeftBound15881.bound (LeftBound15881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound31499.bound LeftBound15881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31499.bound, LeftBound15881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound31499.actual selector witness) * (LeftBound15881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31506

namespace LeftBound31507
def owner : Owner := ⟨.program ⟨257⟩, ⟨17514⟩⟩
def transferEvent : Nat := 31507
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7171⟩⟩]⟩ [⟨.result 15878 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15878 .coefficient)
      LeftAuthority15877.bound (LeftAuthority15877.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7171⟩⟩) (rawTerms := some (Proof.Events062.exact15878RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15877.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15877.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15877.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15877.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound31507

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
