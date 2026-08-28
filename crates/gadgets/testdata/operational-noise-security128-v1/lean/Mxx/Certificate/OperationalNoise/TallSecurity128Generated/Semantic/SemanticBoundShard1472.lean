import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1422
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1471

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound219459
def owner : Owner := ⟨.program ⟨257⟩, ⟨27768⟩⟩
def transferEvent : Nat := 219459
def frameStart : Nat := 219394
def rule : BoundRule := .product (.predecessor 0 219457 .coefficient) (.predecessor 1 219458 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219457 .coefficient)
      LeftAuthority219455.bound (LeftAuthority219455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219458 .coefficient)
      LeftBound219453.bound (LeftBound219453.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219454RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219453.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219453.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority219455.bound LeftBound219453.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219455.bound, LeftBound219453.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority219455.actual selector witness) * (LeftBound219453.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219459

namespace LeftBound219467
def owner : Owner := ⟨.program ⟨257⟩, ⟨27769⟩⟩
def transferEvent : Nat := 219467
def frameStart : Nat := 219394
def rule : BoundRule := .sum [.predecessor 0 219465 .coefficient, .predecessor 1 219466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219465 .coefficient)
      LeftAuthority219463.bound (LeftAuthority219463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219463.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219466 .coefficient)
      LeftBound219459.bound (LeftBound219459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219461RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219459.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority219463.bound, LeftBound219459.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219463.bound, LeftBound219459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority219463.actual selector witness, LeftBound219459.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219467

namespace LeftBound219471
def owner : Owner := ⟨.program ⟨257⟩, ⟨28284⟩⟩
def transferEvent : Nat := 219471
def frameStart : Nat := 219394
def rule : BoundRule := .product (.predecessor 0 219469 .coefficient) (.predecessor 1 219470 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219469 .coefficient)
      LeftBound219467.bound (LeftBound219467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219470 .coefficient)
      LeftAuthority219444.bound (LeftAuthority219444.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219444.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219444.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound219467.bound LeftAuthority219444.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219467.bound, LeftAuthority219444.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound219467.actual selector witness) * (LeftAuthority219444.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219471

namespace LeftBound219482
def owner : Owner := ⟨.program ⟨257⟩, ⟨26624⟩⟩
def transferEvent : Nat := 219482
def frameStart : Nat := 219394
def rule : BoundRule := .product (.predecessor 0 219480 .coefficient) (.predecessor 1 219481 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219480 .coefficient)
      LeftAuthority219455.bound (LeftAuthority219455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219455.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219481 .coefficient)
      LeftAuthority219478.bound (LeftAuthority219478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219478.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority219455.bound LeftAuthority219478.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219455.bound, LeftAuthority219478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority219455.actual selector witness) * (LeftAuthority219478.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219482

namespace LeftBound219490
def owner : Owner := ⟨.program ⟨257⟩, ⟨26625⟩⟩
def transferEvent : Nat := 219490
def frameStart : Nat := 219394
def rule : BoundRule := .sum [.predecessor 0 219488 .coefficient, .predecessor 1 219489 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219488 .coefficient)
      LeftAuthority219486.bound (LeftAuthority219486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219486.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219486.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219489 .coefficient)
      LeftBound219482.bound (LeftBound219482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219482.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219482.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority219486.bound, LeftBound219482.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219486.bound, LeftBound219482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority219486.actual selector witness, LeftBound219482.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219490

namespace LeftBound219494
def owner : Owner := ⟨.program ⟨257⟩, ⟨28288⟩⟩
def transferEvent : Nat := 219494
def frameStart : Nat := 219394
def rule : BoundRule := .sum [.predecessor 0 219492 .coefficient, .predecessor 1 219493 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219492 .coefficient)
      LeftBound219490.bound (LeftBound219490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219490.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219490.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219493 .coefficient)
      LeftBound219471.bound (LeftBound219471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound219490.bound, LeftBound219471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219490.bound, LeftBound219471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound219490.actual selector witness, LeftBound219471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219494

namespace LeftBound219507
def owner : Owner := ⟨.program ⟨257⟩, ⟨28286⟩⟩
def transferEvent : Nat := 219507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 219505 .coefficient, .predecessor 1 219506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219505 .coefficient)
      LeftBound219336.bound (LeftBound219336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219506 .coefficient)
      LeftBound219319.bound (LeftBound219319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219319.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound219336.bound, LeftBound219319.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219336.bound, LeftBound219319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound219336.actual selector witness, LeftBound219319.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219507

namespace LeftBound219510
def owner : Owner := ⟨.program ⟨257⟩, ⟨28286⟩⟩
def transferEvent : Nat := 219510
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 219504 .summary, .result 219326 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219504 .summary)
      LeftBound219338.bound (LeftBound219338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27155⟩⟩) (rawTerms := some (Proof.Events857.exact219504RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219326 .summary)
      LeftBound219321.bound (LeftBound219321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28285⟩⟩) (rawTerms := some (Proof.Events856.exact219326RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound219338.bound, LeftBound219321.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219338.bound, LeftBound219321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound219338.actual selector witness, LeftBound219321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219510

namespace LeftBound219514
def owner : Owner := ⟨.program ⟨257⟩, ⟨28287⟩⟩
def transferEvent : Nat := 219514
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 219512 .coefficient) (.predecessor 1 219513 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219512 .coefficient)
      LeftBound219507.bound (LeftBound219507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219511RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219513 .coefficient)
      LeftBound15681.bound (LeftBound15681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound219507.bound LeftBound15681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219507.bound, LeftBound15681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound219507.actual selector witness) * (LeftBound15681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219514

namespace LeftBound219515
def owner : Owner := ⟨.program ⟨257⟩, ⟨28287⟩⟩
def transferEvent : Nat := 219515
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩ [⟨.result 15678 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15678 .coefficient)
      LeftAuthority15677.bound (LeftAuthority15677.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7169⟩⟩) (rawTerms := some (Proof.Events061.exact15678RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15677.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15677.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15677.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound219515

namespace LeftBound219516
def owner : Owner := ⟨.program ⟨257⟩, ⟨28287⟩⟩
def transferEvent : Nat := 219516
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 219511 .summary) (.transfer 219515) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219511 .summary)
      LeftBound219510.bound (LeftBound219510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28286⟩⟩) (rawTerms := some (Proof.Events857.exact219511RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 219515)
      LeftBound219515.bound (LeftBound219515.actual selector witness) := by
  exact .transfer (LeftBound219515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound219510.bound LeftBound219515.bound
def bound : CoeffClass := .finite ⟨345654216875549026890382321864211871825920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219510.bound, LeftBound219515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound219510.actual selector witness) * (LeftBound219515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219516

namespace LeftBound219531
def owner : Owner := ⟨.program ⟨257⟩, ⟨70164⟩⟩
def transferEvent : Nat := 219531
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 219529 .coefficient) (.predecessor 1 219530 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219529 .coefficient)
      LeftBound211658.bound (LeftBound211658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events826.exact211662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound211658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound211658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219530 .coefficient)
      LeftAuthority219527.bound (LeftAuthority219527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219527.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound211658.bound LeftAuthority219527.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211658.bound, LeftAuthority219527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound211658.actual selector witness) * (LeftAuthority219527.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219531

namespace LeftBound219532
def owner : Owner := ⟨.program ⟨257⟩, ⟨70164⟩⟩
def transferEvent : Nat := 219532
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70162⟩⟩]⟩ [⟨.result 219528 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219528 .coefficient)
      LeftAuthority219527.bound (LeftAuthority219527.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70162⟩⟩) (rawTerms := some (Proof.Events857.exact219528RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219527.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219527.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority219527.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority219527.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound219532

namespace LeftBound219533
def owner : Owner := ⟨.program ⟨257⟩, ⟨70164⟩⟩
def transferEvent : Nat := 219533
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 211662 .summary) (.transfer 219532) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 211662 .summary)
      LeftBound211661.bound (LeftBound211661.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69242⟩⟩) (rawTerms := some (Proof.Events826.exact211662RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound211661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 219532)
      LeftBound219532.bound (LeftBound219532.actual selector witness) := by
  exact .transfer (LeftBound219532.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound211661.bound LeftBound219532.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound211661.bound, LeftBound219532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound211661.actual selector witness) * (LeftBound219532.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219533

namespace LeftBound219544
def owner : Owner := ⟨.program ⟨257⟩, ⟨68075⟩⟩
def transferEvent : Nat := 219544
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 219542 .coefficient) (.value (.predecessor 1 219543 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219542 .coefficient)
      LeftAuthority219540.bound (LeftAuthority219540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219543 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority219540.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219540.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority219540.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound219544

namespace LeftBound219548
def owner : Owner := ⟨.program ⟨257⟩, ⟨68076⟩⟩
def transferEvent : Nat := 219548
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 219546 .coefficient) (.predecessor 1 219547 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219546 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219547 .coefficient)
      LeftBound219544.bound (LeftBound219544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events857.exact219545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219544.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound219544.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound219544.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound219544.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219548

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
