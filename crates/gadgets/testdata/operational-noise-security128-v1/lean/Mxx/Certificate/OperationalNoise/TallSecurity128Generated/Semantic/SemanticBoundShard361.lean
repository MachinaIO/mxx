import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard360

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound59305
def owner : Owner := ⟨.program ⟨257⟩, ⟨57874⟩⟩
def transferEvent : Nat := 59305
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 59303 .coefficient) (.value (.predecessor 1 59304 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59303 .coefficient)
      LeftAuthority59301.bound (LeftAuthority59301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59304 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority59301.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59301.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority59301.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound59305

namespace LeftBound59309
def owner : Owner := ⟨.program ⟨257⟩, ⟨57875⟩⟩
def transferEvent : Nat := 59309
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59307 .coefficient) (.predecessor 1 59308 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59307 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59308 .coefficient)
      LeftBound59305.bound (LeftBound59305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59305.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound59305.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound59305.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound59305.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59309

namespace LeftBound59310
def owner : Owner := ⟨.program ⟨257⟩, ⟨57875⟩⟩
def transferEvent : Nat := 59310
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨57872⟩⟩]⟩ [⟨.result 59302 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59302 .coefficient)
      LeftAuthority59301.bound (LeftAuthority59301.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨57872⟩⟩) (rawTerms := some (Proof.Events231.exact59302RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59301.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59301.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority59301.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority59301.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound59310

namespace LeftBound59311
def owner : Owner := ⟨.program ⟨257⟩, ⟨57875⟩⟩
def transferEvent : Nat := 59311
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 59310) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 59310)
      LeftBound59310.bound (LeftBound59310.actual selector witness) := by
  exact .transfer (LeftBound59310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound59310.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound59310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound59310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59311

namespace LeftBound59406
def owner : Owner := ⟨.program ⟨257⟩, ⟨56913⟩⟩
def transferEvent : Nat := 59406
def frameStart : Nat := 59367
def rule : BoundRule := .identity (.predecessor 0 59405 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59405 .coefficient)
      LeftAuthority59403.bound (LeftAuthority59403.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59403.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59403.derived selector witness)

def rawBound : CoeffClass := LeftAuthority59403.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority59403.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59406

namespace LeftBound59423
def owner : Owner := ⟨.program ⟨257⟩, ⟨58358⟩⟩
def transferEvent : Nat := 59423
def frameStart : Nat := 59367
def rule : BoundRule := .sum [.predecessor 0 59421 .coefficient, .predecessor 1 59422 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59421 .coefficient)
      LeftBound59406.bound (LeftBound59406.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59422 .coefficient)
      LeftAuthority59419.bound (LeftAuthority59419.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority59419.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59406.bound, LeftAuthority59419.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59406.bound, LeftAuthority59419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59406.actual selector witness, LeftAuthority59419.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59423

namespace LeftBound59426
def owner : Owner := ⟨.program ⟨257⟩, ⟨58359⟩⟩
def transferEvent : Nat := 59426
def frameStart : Nat := 59367
def rule : BoundRule := .identity (.predecessor 0 59425 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59425 .coefficient)
      LeftBound59423.bound (LeftBound59423.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound59423.derived selector witness)

def rawBound : CoeffClass := LeftBound59423.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound59423.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound59426

namespace LeftBound59432
def owner : Owner := ⟨.program ⟨257⟩, ⟨58360⟩⟩
def transferEvent : Nat := 59432
def frameStart : Nat := 59367
def rule : BoundRule := .product (.predecessor 0 59430 .coefficient) (.predecessor 1 59431 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59430 .coefficient)
      LeftAuthority59428.bound (LeftAuthority59428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59431 .coefficient)
      LeftBound59426.bound (LeftBound59426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority59428.bound LeftBound59426.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59428.bound, LeftBound59426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority59428.actual selector witness) * (LeftBound59426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59432

namespace LeftBound59440
def owner : Owner := ⟨.program ⟨257⟩, ⟨58361⟩⟩
def transferEvent : Nat := 59440
def frameStart : Nat := 59367
def rule : BoundRule := .sum [.predecessor 0 59438 .coefficient, .predecessor 1 59439 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59438 .coefficient)
      LeftAuthority59436.bound (LeftAuthority59436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59436.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59439 .coefficient)
      LeftBound59432.bound (LeftBound59432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59434RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59436.bound, LeftBound59432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59436.bound, LeftBound59432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority59436.actual selector witness, LeftBound59432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59440

namespace LeftBound59444
def owner : Owner := ⟨.program ⟨257⟩, ⟨59154⟩⟩
def transferEvent : Nat := 59444
def frameStart : Nat := 59367
def rule : BoundRule := .product (.predecessor 0 59442 .coefficient) (.predecessor 1 59443 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59442 .coefficient)
      LeftBound59440.bound (LeftBound59440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59440.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59443 .coefficient)
      LeftAuthority59417.bound (LeftAuthority59417.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59417.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59417.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound59440.bound LeftAuthority59417.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59440.bound, LeftAuthority59417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound59440.actual selector witness) * (LeftAuthority59417.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59444

namespace LeftBound59455
def owner : Owner := ⟨.program ⟨257⟩, ⟨57280⟩⟩
def transferEvent : Nat := 59455
def frameStart : Nat := 59367
def rule : BoundRule := .product (.predecessor 0 59453 .coefficient) (.predecessor 1 59454 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59453 .coefficient)
      LeftAuthority59428.bound (LeftAuthority59428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59454 .coefficient)
      LeftAuthority59451.bound (LeftAuthority59451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59451.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority59428.bound LeftAuthority59451.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59428.bound, LeftAuthority59451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority59428.actual selector witness) * (LeftAuthority59451.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59455

namespace LeftBound59463
def owner : Owner := ⟨.program ⟨257⟩, ⟨57281⟩⟩
def transferEvent : Nat := 59463
def frameStart : Nat := 59367
def rule : BoundRule := .sum [.predecessor 0 59461 .coefficient, .predecessor 1 59462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59461 .coefficient)
      LeftAuthority59459.bound (LeftAuthority59459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority59459.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority59459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59462 .coefficient)
      LeftBound59455.bound (LeftBound59455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59455.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority59459.bound, LeftBound59455.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority59459.bound, LeftBound59455.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority59459.actual selector witness, LeftBound59455.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59463

namespace LeftBound59467
def owner : Owner := ⟨.program ⟨257⟩, ⟨59159⟩⟩
def transferEvent : Nat := 59467
def frameStart : Nat := 59367
def rule : BoundRule := .sum [.predecessor 0 59465 .coefficient, .predecessor 1 59466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59465 .coefficient)
      LeftBound59463.bound (LeftBound59463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59466 .coefficient)
      LeftBound59444.bound (LeftBound59444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59444.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59444.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59463.bound, LeftBound59444.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59463.bound, LeftBound59444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59463.actual selector witness, LeftBound59444.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59467

namespace LeftBound59480
def owner : Owner := ⟨.program ⟨257⟩, ⟨59156⟩⟩
def transferEvent : Nat := 59480
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 59478 .coefficient, .predecessor 1 59479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59478 .coefficient)
      LeftBound59309.bound (LeftBound59309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59479 .coefficient)
      LeftBound59292.bound (LeftBound59292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events231.exact59299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59309.bound, LeftBound59292.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59309.bound, LeftBound59292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59309.actual selector witness, LeftBound59292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59480

namespace LeftBound59483
def owner : Owner := ⟨.program ⟨257⟩, ⟨59156⟩⟩
def transferEvent : Nat := 59483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 59477 .summary, .result 59299 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59477 .summary)
      LeftBound59311.bound (LeftBound59311.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57875⟩⟩) (rawTerms := some (Proof.Events232.exact59477RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 59299 .summary)
      LeftBound59294.bound (LeftBound59294.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59155⟩⟩) (rawTerms := some (Proof.Events231.exact59299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59294.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound59311.bound, LeftBound59294.bound]
def bound : CoeffClass := .finite ⟨32190182365603518530196853751808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59311.bound, LeftBound59294.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound59311.actual selector witness, LeftBound59294.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound59483

namespace LeftBound59487
def owner : Owner := ⟨.program ⟨257⟩, ⟨59157⟩⟩
def transferEvent : Nat := 59487
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 59485 .coefficient) (.predecessor 1 59486 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 59485 .coefficient)
      LeftBound59480.bound (LeftBound59480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events232.exact59484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound59480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound59480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 59486 .coefficient)
      LeftBound15761.bound (LeftBound15761.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15761.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15761.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound59480.bound LeftBound15761.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59480.bound, LeftBound15761.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound59480.actual selector witness) * (LeftBound15761.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound59487

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
