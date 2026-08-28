import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1998
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2048
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2049

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound301404
def owner : Owner := ⟨.program ⟨257⟩, ⟨31750⟩⟩
def transferEvent : Nat := 301404
def frameStart : Nat := 301314
def rule : BoundRule := .product (.predecessor 0 301402 .coefficient) (.predecessor 1 301403 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301402 .coefficient)
      LeftAuthority301357.bound (LeftAuthority301357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301357.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301403 .coefficient)
      LeftAuthority301400.bound (LeftAuthority301400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301401RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301400.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301400.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority301357.bound LeftAuthority301400.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301357.bound, LeftAuthority301400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority301357.actual selector witness) * (LeftAuthority301400.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301404

namespace LeftBound301412
def owner : Owner := ⟨.program ⟨257⟩, ⟨31751⟩⟩
def transferEvent : Nat := 301412
def frameStart : Nat := 301314
def rule : BoundRule := .sum [.predecessor 0 301410 .coefficient, .predecessor 1 301411 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301410 .coefficient)
      LeftAuthority301408.bound (LeftAuthority301408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301411 .coefficient)
      LeftBound301404.bound (LeftBound301404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301404.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority301408.bound, LeftBound301404.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301408.bound, LeftBound301404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority301408.actual selector witness, LeftBound301404.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301412

namespace LeftBound301416
def owner : Owner := ⟨.program ⟨257⟩, ⟨33353⟩⟩
def transferEvent : Nat := 301416
def frameStart : Nat := 301314
def rule : BoundRule := .sum [.predecessor 0 301414 .coefficient, .predecessor 1 301415 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301414 .coefficient)
      LeftBound301412.bound (LeftBound301412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301413RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301412.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301415 .coefficient)
      LeftBound301393.bound (LeftBound301393.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301393.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301393.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301412.bound, LeftBound301393.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301412.bound, LeftBound301393.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301412.actual selector witness, LeftBound301393.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301416

namespace LeftBound301429
def owner : Owner := ⟨.program ⟨257⟩, ⟨33351⟩⟩
def transferEvent : Nat := 301429
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301427 .coefficient, .predecessor 1 301428 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301427 .coefficient)
      LeftBound301274.bound (LeftBound301274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301428 .coefficient)
      LeftBound301257.bound (LeftBound301257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301274.bound, LeftBound301257.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301274.bound, LeftBound301257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301274.actual selector witness, LeftBound301257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301429

namespace LeftBound301432
def owner : Owner := ⟨.program ⟨257⟩, ⟨33351⟩⟩
def transferEvent : Nat := 301432
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 301426 .summary, .result 301264 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301426 .summary)
      LeftBound301276.bound (LeftBound301276.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32292⟩⟩) (rawTerms := some (Proof.Events1177.exact301426RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301264 .summary)
      LeftBound301259.bound (LeftBound301259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33350⟩⟩) (rawTerms := some (Proof.Events1176.exact301264RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301276.bound, LeftBound301259.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301276.bound, LeftBound301259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301276.actual selector witness, LeftBound301259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301432

namespace LeftBound301436
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def transferEvent : Nat := 301436
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301434 .coefficient) (.predecessor 1 301435 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301434 .coefficient)
      LeftBound301429.bound (LeftBound301429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301429.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301435 .coefficient)
      LeftAuthority301179.bound (LeftAuthority301179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301179.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301429.bound LeftAuthority301179.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301429.bound, LeftAuthority301179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301429.actual selector witness) * (LeftAuthority301179.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301436

namespace LeftBound301437
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def transferEvent : Nat := 301437
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33582⟩⟩]⟩ [⟨.result 301180 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301180 .coefficient)
      LeftAuthority301179.bound (LeftAuthority301179.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33582⟩⟩) (rawTerms := some (Proof.Events1176.exact301180RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301179.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority301179.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority301179.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301437

namespace LeftBound301438
def owner : Owner := ⟨.program ⟨257⟩, ⟨33584⟩⟩
def transferEvent : Nat := 301438
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 301433 .summary) (.transfer 301437) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301433 .summary)
      LeftBound301432.bound (LeftBound301432.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33351⟩⟩) (rawTerms := some (Proof.Events1177.exact301433RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 301437)
      LeftBound301437.bound (LeftBound301437.actual selector witness) := by
  exact .transfer (LeftBound301437.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301432.bound LeftBound301437.bound
def bound : CoeffClass := .finite ⟨32189200113374879571150551121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301432.bound, LeftBound301437.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301432.actual selector witness) * (LeftBound301437.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301438

namespace LeftBound301449
def owner : Owner := ⟨.program ⟨257⟩, ⟨32498⟩⟩
def transferEvent : Nat := 301449
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 301447 .coefficient) (.value (.predecessor 1 301448 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301447 .coefficient)
      LeftAuthority301445.bound (LeftAuthority301445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301448 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority301445.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301445.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority301445.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound301449

namespace LeftBound301453
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def transferEvent : Nat := 301453
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301451 .coefficient) (.predecessor 1 301452 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301451 .coefficient)
      LeftBound295192.bound (LeftBound295192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301452 .coefficient)
      LeftBound301449.bound (LeftBound301449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295192.bound LeftBound301449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295192.bound, LeftBound301449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295192.actual selector witness) * (LeftBound301449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301453

namespace LeftBound301454
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def transferEvent : Nat := 301454
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32496⟩⟩]⟩ [⟨.result 301446 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301446 .coefficient)
      LeftAuthority301445.bound (LeftAuthority301445.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32496⟩⟩) (rawTerms := some (Proof.Events1177.exact301446RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301445.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301445.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority301445.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority301445.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301454

namespace LeftBound301455
def owner : Owner := ⟨.program ⟨257⟩, ⟨32499⟩⟩
def transferEvent : Nat := 301455
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 295195 .summary) (.transfer 301454) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295195 .summary)
      LeftBound295193.bound (LeftBound295193.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨2380⟩⟩) (rawTerms := some (Proof.Events1153.exact295195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 301454)
      LeftBound301454.bound (LeftBound301454.actual selector witness) := by
  exact .transfer (LeftBound301454.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound295193.bound LeftBound301454.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound295193.bound, LeftBound301454.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound295193.actual selector witness) * (LeftBound301454.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301455

namespace LeftBound301526
def owner : Owner := ⟨.program ⟨257⟩, ⟨31749⟩⟩
def transferEvent : Nat := 301526
def frameStart : Nat := 301499
def rule : BoundRule := .identity (.predecessor 0 301525 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301525 .coefficient)
      LeftAuthority301523.bound (LeftAuthority301523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301523.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301523.derived selector witness)

def rawBound : CoeffClass := LeftAuthority301523.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority301523.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound301526

namespace LeftBound301543
def owner : Owner := ⟨.program ⟨257⟩, ⟨33266⟩⟩
def transferEvent : Nat := 301543
def frameStart : Nat := 301499
def rule : BoundRule := .sum [.predecessor 0 301541 .coefficient, .predecessor 1 301542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301541 .coefficient)
      LeftBound301526.bound (LeftBound301526.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound301526.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301542 .coefficient)
      LeftAuthority301539.bound (LeftAuthority301539.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority301539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301526.bound, LeftAuthority301539.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301526.bound, LeftAuthority301539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301526.actual selector witness, LeftAuthority301539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301543

namespace LeftBound301546
def owner : Owner := ⟨.program ⟨257⟩, ⟨33267⟩⟩
def transferEvent : Nat := 301546
def frameStart : Nat := 301499
def rule : BoundRule := .identity (.predecessor 0 301545 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301545 .coefficient)
      LeftBound301543.bound (LeftBound301543.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound301543.derived selector witness)

def rawBound : CoeffClass := LeftBound301543.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound301543.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound301546

namespace LeftBound301552
def owner : Owner := ⟨.program ⟨257⟩, ⟨33268⟩⟩
def transferEvent : Nat := 301552
def frameStart : Nat := 301499
def rule : BoundRule := .product (.predecessor 0 301550 .coefficient) (.predecessor 1 301551 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301550 .coefficient)
      LeftAuthority301548.bound (LeftAuthority301548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301549RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301548.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301548.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301551 .coefficient)
      LeftBound301546.bound (LeftBound301546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1177.exact301547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301546.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority301548.bound LeftBound301546.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority301548.bound, LeftBound301546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority301548.actual selector witness) * (LeftBound301546.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301552

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
