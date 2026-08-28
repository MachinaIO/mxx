import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard784
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard788
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard854

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound130299
def owner : Owner := ⟨.program ⟨257⟩, ⟨47245⟩⟩
def transferEvent : Nat := 130299
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 120538 .summary) (.transfer 130298) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 120538 .summary)
      LeftBound120537.bound (LeftBound120537.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46937⟩⟩) (rawTerms := some (Proof.Events470.exact120538RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound120537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 130298)
      LeftBound130298.bound (LeftBound130298.actual selector witness) := by
  exact .transfer (LeftBound130298.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound120537.bound LeftBound130298.bound
def bound : CoeffClass := .finite ⟨32194307824962751379413684715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound120537.bound, LeftBound130298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound120537.actual selector witness) * (LeftBound130298.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130299

namespace LeftBound130310
def owner : Owner := ⟨.program ⟨257⟩, ⟨46134⟩⟩
def transferEvent : Nat := 130310
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 130308 .coefficient) (.value (.predecessor 1 130309 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130308 .coefficient)
      LeftAuthority130306.bound (LeftAuthority130306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130309 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority130306.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130306.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority130306.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound130310

namespace LeftBound130314
def owner : Owner := ⟨.program ⟨257⟩, ⟨46135⟩⟩
def transferEvent : Nat := 130314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 130312 .coefficient) (.predecessor 1 130313 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130312 .coefficient)
      LeftBound119867.bound (LeftBound119867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130313 .coefficient)
      LeftBound130310.bound (LeftBound130310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119867.bound LeftBound130310.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119867.bound, LeftBound130310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119867.actual selector witness) * (LeftBound130310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130314

namespace LeftBound130315
def owner : Owner := ⟨.program ⟨257⟩, ⟨46135⟩⟩
def transferEvent : Nat := 130315
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨46132⟩⟩]⟩ [⟨.result 130307 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130307 .coefficient)
      LeftAuthority130306.bound (LeftAuthority130306.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨46132⟩⟩) (rawTerms := some (Proof.Events509.exact130307RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130306.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority130306.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority130306.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound130315

namespace LeftBound130316
def owner : Owner := ⟨.program ⟨257⟩, ⟨46135⟩⟩
def transferEvent : Nat := 130316
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 119870 .summary) (.transfer 130315) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119870 .summary)
      LeftBound119868.bound (LeftBound119868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5527⟩⟩) (rawTerms := some (Proof.Events468.exact119870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 130315)
      LeftBound130315.bound (LeftBound130315.actual selector witness) := by
  exact .transfer (LeftBound130315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound119868.bound LeftBound130315.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119868.bound, LeftBound130315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound119868.actual selector witness) * (LeftBound130315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130316

namespace LeftBound130411
def owner : Owner := ⟨.program ⟨257⟩, ⟨45437⟩⟩
def transferEvent : Nat := 130411
def frameStart : Nat := 130372
def rule : BoundRule := .identity (.predecessor 0 130410 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130410 .coefficient)
      LeftAuthority130408.bound (LeftAuthority130408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130408.derived selector witness)

def rawBound : CoeffClass := LeftAuthority130408.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority130408.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound130411

namespace LeftBound130428
def owner : Owner := ⟨.program ⟨257⟩, ⟨46810⟩⟩
def transferEvent : Nat := 130428
def frameStart : Nat := 130372
def rule : BoundRule := .sum [.predecessor 0 130426 .coefficient, .predecessor 1 130427 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130426 .coefficient)
      LeftBound130411.bound (LeftBound130411.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound130411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130427 .coefficient)
      LeftAuthority130424.bound (LeftAuthority130424.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority130424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130411.bound, LeftAuthority130424.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130411.bound, LeftAuthority130424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130411.actual selector witness, LeftAuthority130424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130428

namespace LeftBound130431
def owner : Owner := ⟨.program ⟨257⟩, ⟨46811⟩⟩
def transferEvent : Nat := 130431
def frameStart : Nat := 130372
def rule : BoundRule := .identity (.predecessor 0 130430 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130430 .coefficient)
      LeftBound130428.bound (LeftBound130428.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound130428.derived selector witness)

def rawBound : CoeffClass := LeftBound130428.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound130428.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound130431

namespace LeftBound130437
def owner : Owner := ⟨.program ⟨257⟩, ⟨46812⟩⟩
def transferEvent : Nat := 130437
def frameStart : Nat := 130372
def rule : BoundRule := .product (.predecessor 0 130435 .coefficient) (.predecessor 1 130436 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130435 .coefficient)
      LeftAuthority130433.bound (LeftAuthority130433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130436 .coefficient)
      LeftBound130431.bound (LeftBound130431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority130433.bound LeftBound130431.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130433.bound, LeftBound130431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority130433.actual selector witness) * (LeftBound130431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130437

namespace LeftBound130445
def owner : Owner := ⟨.program ⟨257⟩, ⟨46813⟩⟩
def transferEvent : Nat := 130445
def frameStart : Nat := 130372
def rule : BoundRule := .sum [.predecessor 0 130443 .coefficient, .predecessor 1 130444 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130443 .coefficient)
      LeftAuthority130441.bound (LeftAuthority130441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130444 .coefficient)
      LeftBound130437.bound (LeftBound130437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130437.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130437.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority130441.bound, LeftBound130437.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130441.bound, LeftBound130437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority130441.actual selector witness, LeftBound130437.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130445

namespace LeftBound130449
def owner : Owner := ⟨.program ⟨257⟩, ⟨47244⟩⟩
def transferEvent : Nat := 130449
def frameStart : Nat := 130372
def rule : BoundRule := .product (.predecessor 0 130447 .coefficient) (.predecessor 1 130448 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130447 .coefficient)
      LeftBound130445.bound (LeftBound130445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130448 .coefficient)
      LeftAuthority130422.bound (LeftAuthority130422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130422.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound130445.bound LeftAuthority130422.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130445.bound, LeftAuthority130422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound130445.actual selector witness) * (LeftAuthority130422.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130449

namespace LeftBound130460
def owner : Owner := ⟨.program ⟨257⟩, ⟨45629⟩⟩
def transferEvent : Nat := 130460
def frameStart : Nat := 130372
def rule : BoundRule := .product (.predecessor 0 130458 .coefficient) (.predecessor 1 130459 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130458 .coefficient)
      LeftAuthority130433.bound (LeftAuthority130433.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130433.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130433.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130459 .coefficient)
      LeftAuthority130456.bound (LeftAuthority130456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority130433.bound LeftAuthority130456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130433.bound, LeftAuthority130456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority130433.actual selector witness) * (LeftAuthority130456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound130460

namespace LeftBound130468
def owner : Owner := ⟨.program ⟨257⟩, ⟨45630⟩⟩
def transferEvent : Nat := 130468
def frameStart : Nat := 130372
def rule : BoundRule := .sum [.predecessor 0 130466 .coefficient, .predecessor 1 130467 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130466 .coefficient)
      LeftAuthority130464.bound (LeftAuthority130464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority130464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority130464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130467 .coefficient)
      LeftBound130460.bound (LeftBound130460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority130464.bound, LeftBound130460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority130464.bound, LeftBound130460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority130464.actual selector witness, LeftBound130460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130468

namespace LeftBound130472
def owner : Owner := ⟨.program ⟨257⟩, ⟨47248⟩⟩
def transferEvent : Nat := 130472
def frameStart : Nat := 130372
def rule : BoundRule := .sum [.predecessor 0 130470 .coefficient, .predecessor 1 130471 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130470 .coefficient)
      LeftBound130468.bound (LeftBound130468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130471 .coefficient)
      LeftBound130449.bound (LeftBound130449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130468.bound, LeftBound130449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130468.bound, LeftBound130449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130468.actual selector witness, LeftBound130449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130472

namespace LeftBound130485
def owner : Owner := ⟨.program ⟨257⟩, ⟨47246⟩⟩
def transferEvent : Nat := 130485
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 130483 .coefficient, .predecessor 1 130484 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 130483 .coefficient)
      LeftBound130314.bound (LeftBound130314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 130484 .coefficient)
      LeftBound130297.bound (LeftBound130297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events509.exact130304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound130297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound130297.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130314.bound, LeftBound130297.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130314.bound, LeftBound130297.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130314.actual selector witness, LeftBound130297.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130485

namespace LeftBound130488
def owner : Owner := ⟨.program ⟨257⟩, ⟨47246⟩⟩
def transferEvent : Nat := 130488
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 130482 .summary, .result 130304 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130482 .summary)
      LeftBound130316.bound (LeftBound130316.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46135⟩⟩) (rawTerms := some (Proof.Events509.exact130482RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 130304 .summary)
      LeftBound130299.bound (LeftBound130299.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47245⟩⟩) (rawTerms := some (Proof.Events509.exact130304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound130299.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound130316.bound, LeftBound130299.bound]
def bound : CoeffClass := .finite ⟨32194307824962953452255538577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound130316.bound, LeftBound130299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound130316.actual selector witness, LeftBound130299.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound130488

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
