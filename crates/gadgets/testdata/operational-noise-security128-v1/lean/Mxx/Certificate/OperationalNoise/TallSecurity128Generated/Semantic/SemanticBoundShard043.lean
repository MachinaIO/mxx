import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard041
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard042

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound13459
def owner : Owner := ⟨.program ⟨257⟩, ⟨31946⟩⟩
def transferEvent : Nat := 13459
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13457 .coefficient, .predecessor 1 13458 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13457 .coefficient)
      LeftBound13455.bound (LeftBound13455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13458 .coefficient)
      LeftBound13418.bound (LeftBound13418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13418.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13455.bound, LeftBound13418.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13455.bound, LeftBound13418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13455.actual selector witness, LeftBound13418.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13459

namespace LeftBound13463
def owner : Owner := ⟨.program ⟨257⟩, ⟨51010⟩⟩
def transferEvent : Nat := 13463
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13461 .coefficient, .predecessor 1 13462 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13461 .coefficient)
      LeftBound13459.bound (LeftBound13459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13462 .coefficient)
      LeftBound13410.bound (LeftBound13410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13410.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13410.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13459.bound, LeftBound13410.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13459.bound, LeftBound13410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13459.actual selector witness, LeftBound13410.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13463

namespace LeftBound13467
def owner : Owner := ⟨.program ⟨257⟩, ⟨53990⟩⟩
def transferEvent : Nat := 13467
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13465 .coefficient, .predecessor 1 13466 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13465 .coefficient)
      LeftBound13463.bound (LeftBound13463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13463.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13463.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13466 .coefficient)
      LeftBound13402.bound (LeftBound13402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13463.bound, LeftBound13402.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13463.bound, LeftBound13402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13463.actual selector witness, LeftBound13402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13467

namespace LeftBound13471
def owner : Owner := ⟨.program ⟨257⟩, ⟨56970⟩⟩
def transferEvent : Nat := 13471
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13469 .coefficient, .predecessor 1 13470 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13469 .coefficient)
      LeftBound13467.bound (LeftBound13467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13470 .coefficient)
      LeftBound13394.bound (LeftBound13394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13394.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13467.bound, LeftBound13394.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13467.bound, LeftBound13394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13467.actual selector witness, LeftBound13394.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13471

namespace LeftBound13475
def owner : Owner := ⟨.program ⟨257⟩, ⟨59950⟩⟩
def transferEvent : Nat := 13475
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13473 .coefficient, .predecessor 1 13474 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13473 .coefficient)
      LeftBound13471.bound (LeftBound13471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13471.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13474 .coefficient)
      LeftBound13386.bound (LeftBound13386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13386.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13471.bound, LeftBound13386.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13471.bound, LeftBound13386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13471.actual selector witness, LeftBound13386.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13475

namespace LeftBound13479
def owner : Owner := ⟨.program ⟨257⟩, ⟨62930⟩⟩
def transferEvent : Nat := 13479
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13477 .coefficient, .predecessor 1 13478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13477 .coefficient)
      LeftBound13475.bound (LeftBound13475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13478 .coefficient)
      LeftBound13378.bound (LeftBound13378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13475.bound, LeftBound13378.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13475.bound, LeftBound13378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13475.actual selector witness, LeftBound13378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13479

namespace LeftBound13483
def owner : Owner := ⟨.program ⟨257⟩, ⟨66008⟩⟩
def transferEvent : Nat := 13483
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13481 .coefficient, .predecessor 1 13482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13481 .coefficient)
      LeftBound13479.bound (LeftBound13479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13482 .coefficient)
      LeftBound13370.bound (LeftBound13370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13479.bound, LeftBound13370.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13479.bound, LeftBound13370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13479.actual selector witness, LeftBound13370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13483

namespace LeftBound13487
def owner : Owner := ⟨.program ⟨257⟩, ⟨66009⟩⟩
def transferEvent : Nat := 13487
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13485 .coefficient, .predecessor 1 13486 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13485 .coefficient)
      LeftBound13483.bound (LeftBound13483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13483.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13486 .coefficient)
      LeftBound13362.bound (LeftBound13362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13483.bound, LeftBound13362.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13483.bound, LeftBound13362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13483.actual selector witness, LeftBound13362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13487

namespace LeftBound13491
def owner : Owner := ⟨.program ⟨257⟩, ⟨66010⟩⟩
def transferEvent : Nat := 13491
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13489 .coefficient, .predecessor 1 13490 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13489 .coefficient)
      LeftBound13487.bound (LeftBound13487.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13488RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13487.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13487.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13490 .coefficient)
      LeftBound13354.bound (LeftBound13354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13487.bound, LeftBound13354.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13487.bound, LeftBound13354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13487.actual selector witness, LeftBound13354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13491

namespace LeftBound13495
def owner : Owner := ⟨.program ⟨257⟩, ⟨66011⟩⟩
def transferEvent : Nat := 13495
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13493 .coefficient, .predecessor 1 13494 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13493 .coefficient)
      LeftBound13491.bound (LeftBound13491.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13492RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13491.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13491.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13494 .coefficient)
      LeftBound13346.bound (LeftBound13346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13491.bound, LeftBound13346.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13491.bound, LeftBound13346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13491.actual selector witness, LeftBound13346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13495

namespace LeftBound13499
def owner : Owner := ⟨.program ⟨257⟩, ⟨66012⟩⟩
def transferEvent : Nat := 13499
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13497 .coefficient, .predecessor 1 13498 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13497 .coefficient)
      LeftBound13495.bound (LeftBound13495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13495.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13498 .coefficient)
      LeftBound13338.bound (LeftBound13338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13495.bound, LeftBound13338.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13495.bound, LeftBound13338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13495.actual selector witness, LeftBound13338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13499

namespace LeftBound13503
def owner : Owner := ⟨.program ⟨257⟩, ⟨66013⟩⟩
def transferEvent : Nat := 13503
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13501 .coefficient, .predecessor 1 13502 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13501 .coefficient)
      LeftBound13499.bound (LeftBound13499.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13499.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13499.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13502 .coefficient)
      LeftBound13330.bound (LeftBound13330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13330.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13499.bound, LeftBound13330.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13499.bound, LeftBound13330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13499.actual selector witness, LeftBound13330.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13503

namespace LeftBound13507
def owner : Owner := ⟨.program ⟨257⟩, ⟨66014⟩⟩
def transferEvent : Nat := 13507
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13505 .coefficient, .predecessor 1 13506 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13505 .coefficient)
      LeftBound13503.bound (LeftBound13503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13503.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13506 .coefficient)
      LeftBound13322.bound (LeftBound13322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13322.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13503.bound, LeftBound13322.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13503.bound, LeftBound13322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13503.actual selector witness, LeftBound13322.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13507

namespace LeftBound13511
def owner : Owner := ⟨.program ⟨257⟩, ⟨66015⟩⟩
def transferEvent : Nat := 13511
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13509 .coefficient, .predecessor 1 13510 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13509 .coefficient)
      LeftBound13507.bound (LeftBound13507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13507.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13510 .coefficient)
      LeftBound13314.bound (LeftBound13314.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13316RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13314.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13314.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13507.bound, LeftBound13314.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13507.bound, LeftBound13314.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13507.actual selector witness, LeftBound13314.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13511

namespace LeftBound13515
def owner : Owner := ⟨.program ⟨257⟩, ⟨66016⟩⟩
def transferEvent : Nat := 13515
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13513 .coefficient, .predecessor 1 13514 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13513 .coefficient)
      LeftBound13511.bound (LeftBound13511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13511.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13514 .coefficient)
      LeftBound13306.bound (LeftBound13306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13306.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13511.bound, LeftBound13306.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13511.bound, LeftBound13306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13511.actual selector witness, LeftBound13306.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13515

namespace LeftBound13519
def owner : Owner := ⟨.program ⟨257⟩, ⟨67303⟩⟩
def transferEvent : Nat := 13519
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 13517 .coefficient, .predecessor 1 13518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13517 .coefficient)
      LeftBound13515.bound (LeftBound13515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13518 .coefficient)
      LeftBound13298.bound (LeftBound13298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13298.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound13515.bound, LeftBound13298.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13515.bound, LeftBound13298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound13515.actual selector witness, LeftBound13298.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound13519

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
