import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1458

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound216368
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def transferEvent : Nat := 216368
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 216366 .coefficient) (.predecessor 1 216367 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216366 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216367 .coefficient)
      LeftBound216364.bound (LeftBound216364.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events845.exact216365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216364.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216364.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound216364.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound216364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound216364.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216368

namespace LeftBound216369
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def transferEvent : Nat := 216369
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68370⟩⟩]⟩ [⟨.result 216361 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216361 .coefficient)
      LeftAuthority216360.bound (LeftAuthority216360.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68370⟩⟩) (rawTerms := some (Proof.Events845.exact216361RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216360.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority216360.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority216360.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound216369

namespace LeftBound216370
def owner : Owner := ⟨.program ⟨257⟩, ⟨68373⟩⟩
def transferEvent : Nat := 216370
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 216369) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 216369)
      LeftBound216369.bound (LeftBound216369.actual selector witness) := by
  exact .transfer (LeftBound216369.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound216369.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound216369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound216369.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216370

namespace LeftBound217398
def owner : Owner := ⟨.program ⟨257⟩, ⟨18867⟩⟩
def transferEvent : Nat := 217398
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217396 .coefficient, .predecessor 1 217397 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217396 .coefficient)
      LeftAuthority217394.bound (LeftAuthority217394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217394.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217394.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217397 .coefficient)
      LeftAuthority217371.bound (LeftAuthority217371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217371.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority217394.bound, LeftAuthority217371.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority217394.bound, LeftAuthority217371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority217394.actual selector witness, LeftAuthority217371.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217398

namespace LeftBound217402
def owner : Owner := ⟨.program ⟨257⟩, ⟨22087⟩⟩
def transferEvent : Nat := 217402
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217400 .coefficient, .predecessor 1 217401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217400 .coefficient)
      LeftBound217398.bound (LeftBound217398.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217398.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217398.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217401 .coefficient)
      LeftAuthority217348.bound (LeftAuthority217348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217348.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217398.bound, LeftAuthority217348.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217398.bound, LeftAuthority217348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217398.actual selector witness, LeftAuthority217348.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217402

namespace LeftBound217406
def owner : Owner := ⟨.program ⟨257⟩, ⟨32107⟩⟩
def transferEvent : Nat := 217406
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217404 .coefficient, .predecessor 1 217405 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217404 .coefficient)
      LeftBound217402.bound (LeftBound217402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217405 .coefficient)
      LeftAuthority217325.bound (LeftAuthority217325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217325.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217402.bound, LeftAuthority217325.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217402.bound, LeftAuthority217325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217402.actual selector witness, LeftAuthority217325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217406

namespace LeftBound217410
def owner : Owner := ⟨.program ⟨257⟩, ⟨51162⟩⟩
def transferEvent : Nat := 217410
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217408 .coefficient, .predecessor 1 217409 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217408 .coefficient)
      LeftBound217406.bound (LeftBound217406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217409 .coefficient)
      LeftAuthority217302.bound (LeftAuthority217302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217302.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217406.bound, LeftAuthority217302.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217406.bound, LeftAuthority217302.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217406.actual selector witness, LeftAuthority217302.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217410

namespace LeftBound217414
def owner : Owner := ⟨.program ⟨257⟩, ⟨54142⟩⟩
def transferEvent : Nat := 217414
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217412 .coefficient, .predecessor 1 217413 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217412 .coefficient)
      LeftBound217410.bound (LeftBound217410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217413 .coefficient)
      LeftAuthority217279.bound (LeftAuthority217279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217279.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217410.bound, LeftAuthority217279.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217410.bound, LeftAuthority217279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217410.actual selector witness, LeftAuthority217279.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217414

namespace LeftBound217418
def owner : Owner := ⟨.program ⟨257⟩, ⟨57122⟩⟩
def transferEvent : Nat := 217418
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217416 .coefficient, .predecessor 1 217417 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217416 .coefficient)
      LeftBound217414.bound (LeftBound217414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217415RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217414.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217417 .coefficient)
      LeftAuthority217256.bound (LeftAuthority217256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217256.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217414.bound, LeftAuthority217256.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217414.bound, LeftAuthority217256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217414.actual selector witness, LeftAuthority217256.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217418

namespace LeftBound217422
def owner : Owner := ⟨.program ⟨257⟩, ⟨60102⟩⟩
def transferEvent : Nat := 217422
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217420 .coefficient, .predecessor 1 217421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217420 .coefficient)
      LeftBound217418.bound (LeftBound217418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217421 .coefficient)
      LeftAuthority217233.bound (LeftAuthority217233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217233.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217418.bound, LeftAuthority217233.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217418.bound, LeftAuthority217233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217418.actual selector witness, LeftAuthority217233.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217422

namespace LeftBound217426
def owner : Owner := ⟨.program ⟨257⟩, ⟨63082⟩⟩
def transferEvent : Nat := 217426
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217424 .coefficient, .predecessor 1 217425 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217424 .coefficient)
      LeftBound217422.bound (LeftBound217422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217422.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217425 .coefficient)
      LeftAuthority217210.bound (LeftAuthority217210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217422.bound, LeftAuthority217210.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217422.bound, LeftAuthority217210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217422.actual selector witness, LeftAuthority217210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217426

namespace LeftBound217430
def owner : Owner := ⟨.program ⟨257⟩, ⟨66602⟩⟩
def transferEvent : Nat := 217430
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217428 .coefficient, .predecessor 1 217429 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217428 .coefficient)
      LeftBound217426.bound (LeftBound217426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217429 .coefficient)
      LeftAuthority217187.bound (LeftAuthority217187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217187.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217187.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217426.bound, LeftAuthority217187.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217426.bound, LeftAuthority217187.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217426.actual selector witness, LeftAuthority217187.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217430

namespace LeftBound217434
def owner : Owner := ⟨.program ⟨257⟩, ⟨66603⟩⟩
def transferEvent : Nat := 217434
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217432 .coefficient, .predecessor 1 217433 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217432 .coefficient)
      LeftBound217430.bound (LeftBound217430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217431RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217430.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217433 .coefficient)
      LeftAuthority217164.bound (LeftAuthority217164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217430.bound, LeftAuthority217164.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217430.bound, LeftAuthority217164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217430.actual selector witness, LeftAuthority217164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217434

namespace LeftBound217438
def owner : Owner := ⟨.program ⟨257⟩, ⟨66604⟩⟩
def transferEvent : Nat := 217438
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217436 .coefficient, .predecessor 1 217437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217436 .coefficient)
      LeftBound217434.bound (LeftBound217434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217434.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217437 .coefficient)
      LeftAuthority217141.bound (LeftAuthority217141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217434.bound, LeftAuthority217141.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217434.bound, LeftAuthority217141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217434.actual selector witness, LeftAuthority217141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217438

namespace LeftBound217442
def owner : Owner := ⟨.program ⟨257⟩, ⟨66605⟩⟩
def transferEvent : Nat := 217442
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217440 .coefficient, .predecessor 1 217441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217440 .coefficient)
      LeftBound217438.bound (LeftBound217438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217441 .coefficient)
      LeftAuthority217118.bound (LeftAuthority217118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217438.bound, LeftAuthority217118.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217438.bound, LeftAuthority217118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217438.actual selector witness, LeftAuthority217118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217442

namespace LeftBound217446
def owner : Owner := ⟨.program ⟨257⟩, ⟨66606⟩⟩
def transferEvent : Nat := 217446
def frameStart : Nat := 216961
def rule : BoundRule := .sum [.predecessor 0 217444 .coefficient, .predecessor 1 217445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 217444 .coefficient)
      LeftBound217442.bound (LeftBound217442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events849.exact217443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound217442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound217442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 217445 .coefficient)
      LeftAuthority217095.bound (LeftAuthority217095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events848.exact217096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority217095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority217095.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound217442.bound, LeftAuthority217095.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound217442.bound, LeftAuthority217095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound217442.actual selector witness, LeftAuthority217095.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound217446

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
