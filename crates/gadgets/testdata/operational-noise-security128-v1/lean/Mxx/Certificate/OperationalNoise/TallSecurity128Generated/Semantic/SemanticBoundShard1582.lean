import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1494
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1545
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1546

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound235428
def owner : Owner := ⟨.program ⟨257⟩, ⟨33856⟩⟩
def transferEvent : Nat := 235428
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 235426 .coefficient) (.predecessor 1 235427 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235426 .coefficient)
      LeftBound229175.bound (LeftBound229175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events895.exact229179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound229175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound229175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235427 .coefficient)
      LeftAuthority235424.bound (LeftAuthority235424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events919.exact235425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235424.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound229175.bound LeftAuthority235424.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound229175.bound, LeftAuthority235424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound229175.actual selector witness) * (LeftAuthority235424.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235428

namespace LeftBound235429
def owner : Owner := ⟨.program ⟨257⟩, ⟨33856⟩⟩
def transferEvent : Nat := 235429
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33854⟩⟩]⟩ [⟨.result 235425 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235425 .coefficient)
      LeftAuthority235424.bound (LeftAuthority235424.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33854⟩⟩) (rawTerms := some (Proof.Events919.exact235425RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235424.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority235424.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority235424.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound235429

namespace LeftBound235430
def owner : Owner := ⟨.program ⟨257⟩, ⟨33856⟩⟩
def transferEvent : Nat := 235430
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 229179 .summary) (.transfer 235429) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 229179 .summary)
      LeftBound229178.bound (LeftBound229178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33450⟩⟩) (rawTerms := some (Proof.Events895.exact229179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound229178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 235429)
      LeftBound235429.bound (LeftBound235429.actual selector witness) := by
  exact .transfer (LeftBound235429.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound229178.bound LeftBound235429.bound
def bound : CoeffClass := .finite ⟨32189200113374879571150551121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound229178.bound, LeftBound235429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound229178.actual selector witness) * (LeftBound235429.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235430

namespace LeftBound235441
def owner : Owner := ⟨.program ⟨257⟩, ⟨32674⟩⟩
def transferEvent : Nat := 235441
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 235439 .coefficient) (.value (.predecessor 1 235440 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235439 .coefficient)
      LeftAuthority235437.bound (LeftAuthority235437.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events919.exact235438RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235437.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235440 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority235437.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235437.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority235437.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound235441

namespace LeftBound235445
def owner : Owner := ⟨.program ⟨257⟩, ⟨32675⟩⟩
def transferEvent : Nat := 235445
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 235443 .coefficient) (.predecessor 1 235444 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235443 .coefficient)
      LeftBound222242.bound (LeftBound222242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235444 .coefficient)
      LeftBound235441.bound (LeftBound235441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events919.exact235442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235441.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235441.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222242.bound LeftBound235441.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222242.bound, LeftBound235441.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222242.actual selector witness) * (LeftBound235441.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235445

namespace LeftBound235446
def owner : Owner := ⟨.program ⟨257⟩, ⟨32675⟩⟩
def transferEvent : Nat := 235446
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32672⟩⟩]⟩ [⟨.result 235438 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 235438 .coefficient)
      LeftAuthority235437.bound (LeftAuthority235437.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32672⟩⟩) (rawTerms := some (Proof.Events919.exact235438RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235437.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235437.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority235437.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority235437.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound235446

namespace LeftBound235447
def owner : Owner := ⟨.program ⟨257⟩, ⟨32675⟩⟩
def transferEvent : Nat := 235447
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 222245 .summary) (.transfer 235446) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222245 .summary)
      LeftBound222243.bound (LeftBound222243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5581⟩⟩) (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 235446)
      LeftBound235446.bound (LeftBound235446.actual selector witness) := by
  exact .transfer (LeftBound235446.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222243.bound LeftBound235446.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222243.bound, LeftBound235446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222243.actual selector witness) * (LeftBound235446.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235447

namespace LeftBound235542
def owner : Owner := ⟨.program ⟨257⟩, ⟨31821⟩⟩
def transferEvent : Nat := 235542
def frameStart : Nat := 235503
def rule : BoundRule := .identity (.predecessor 0 235541 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235541 .coefficient)
      LeftAuthority235539.bound (LeftAuthority235539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235539.derived selector witness)

def rawBound : CoeffClass := LeftAuthority235539.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority235539.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound235542

namespace LeftBound235559
def owner : Owner := ⟨.program ⟨257⟩, ⟨33302⟩⟩
def transferEvent : Nat := 235559
def frameStart : Nat := 235503
def rule : BoundRule := .sum [.predecessor 0 235557 .coefficient, .predecessor 1 235558 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235557 .coefficient)
      LeftBound235542.bound (LeftBound235542.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound235542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235558 .coefficient)
      LeftAuthority235555.bound (LeftAuthority235555.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority235555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound235542.bound, LeftAuthority235555.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound235542.bound, LeftAuthority235555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound235542.actual selector witness, LeftAuthority235555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound235559

namespace LeftBound235562
def owner : Owner := ⟨.program ⟨257⟩, ⟨33303⟩⟩
def transferEvent : Nat := 235562
def frameStart : Nat := 235503
def rule : BoundRule := .identity (.predecessor 0 235561 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235561 .coefficient)
      LeftBound235559.bound (LeftBound235559.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound235559.derived selector witness)

def rawBound : CoeffClass := LeftBound235559.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound235559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound235559.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound235562

namespace LeftBound235568
def owner : Owner := ⟨.program ⟨257⟩, ⟨33304⟩⟩
def transferEvent : Nat := 235568
def frameStart : Nat := 235503
def rule : BoundRule := .product (.predecessor 0 235566 .coefficient) (.predecessor 1 235567 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235566 .coefficient)
      LeftAuthority235564.bound (LeftAuthority235564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235567 .coefficient)
      LeftBound235562.bound (LeftBound235562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority235564.bound LeftBound235562.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235564.bound, LeftBound235562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority235564.actual selector witness) * (LeftBound235562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235568

namespace LeftBound235576
def owner : Owner := ⟨.program ⟨257⟩, ⟨33305⟩⟩
def transferEvent : Nat := 235576
def frameStart : Nat := 235503
def rule : BoundRule := .sum [.predecessor 0 235574 .coefficient, .predecessor 1 235575 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235574 .coefficient)
      LeftAuthority235572.bound (LeftAuthority235572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235575 .coefficient)
      LeftBound235568.bound (LeftBound235568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority235572.bound, LeftBound235568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235572.bound, LeftBound235568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority235572.actual selector witness, LeftBound235568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound235576

namespace LeftBound235580
def owner : Owner := ⟨.program ⟨257⟩, ⟨33855⟩⟩
def transferEvent : Nat := 235580
def frameStart : Nat := 235503
def rule : BoundRule := .product (.predecessor 0 235578 .coefficient) (.predecessor 1 235579 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235578 .coefficient)
      LeftBound235576.bound (LeftBound235576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235579 .coefficient)
      LeftAuthority235553.bound (LeftAuthority235553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235553.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound235576.bound LeftAuthority235553.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound235576.bound, LeftAuthority235553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound235576.actual selector witness) * (LeftAuthority235553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235580

namespace LeftBound235591
def owner : Owner := ⟨.program ⟨257⟩, ⟨32085⟩⟩
def transferEvent : Nat := 235591
def frameStart : Nat := 235503
def rule : BoundRule := .product (.predecessor 0 235589 .coefficient) (.predecessor 1 235590 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235589 .coefficient)
      LeftAuthority235564.bound (LeftAuthority235564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235590 .coefficient)
      LeftAuthority235587.bound (LeftAuthority235587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235587.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235587.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority235564.bound LeftAuthority235587.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235564.bound, LeftAuthority235587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority235564.actual selector witness) * (LeftAuthority235587.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound235591

namespace LeftBound235599
def owner : Owner := ⟨.program ⟨257⟩, ⟨32086⟩⟩
def transferEvent : Nat := 235599
def frameStart : Nat := 235503
def rule : BoundRule := .sum [.predecessor 0 235597 .coefficient, .predecessor 1 235598 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235597 .coefficient)
      LeftAuthority235595.bound (LeftAuthority235595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority235595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority235595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235598 .coefficient)
      LeftBound235591.bound (LeftBound235591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235591.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority235595.bound, LeftBound235591.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority235595.bound, LeftBound235591.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority235595.actual selector witness, LeftBound235591.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound235599

namespace LeftBound235603
def owner : Owner := ⟨.program ⟨257⟩, ⟨33860⟩⟩
def transferEvent : Nat := 235603
def frameStart : Nat := 235503
def rule : BoundRule := .sum [.predecessor 0 235601 .coefficient, .predecessor 1 235602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 235601 .coefficient)
      LeftBound235599.bound (LeftBound235599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235599.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 235602 .coefficient)
      LeftBound235580.bound (LeftBound235580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events920.exact235585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound235580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound235580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound235599.bound, LeftBound235580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound235599.bound, LeftBound235580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound235599.actual selector witness, LeftBound235580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound235603

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
