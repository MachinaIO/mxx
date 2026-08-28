import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1966

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound290527
def owner : Owner := ⟨.program ⟨257⟩, ⟨66184⟩⟩
def transferEvent : Nat := 290527
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290525 .coefficient, .predecessor 1 290526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290525 .coefficient)
      LeftBound290523.bound (LeftBound290523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290523.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290526 .coefficient)
      LeftAuthority290230.bound (LeftAuthority290230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290230.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290523.bound, LeftAuthority290230.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290523.bound, LeftAuthority290230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290523.actual selector witness, LeftAuthority290230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290527

namespace LeftBound290531
def owner : Owner := ⟨.program ⟨257⟩, ⟨66185⟩⟩
def transferEvent : Nat := 290531
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290529 .coefficient, .predecessor 1 290530 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290529 .coefficient)
      LeftBound290527.bound (LeftBound290527.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290527.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290530 .coefficient)
      LeftAuthority290207.bound (LeftAuthority290207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290207.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290527.bound, LeftAuthority290207.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290527.bound, LeftAuthority290207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290527.actual selector witness, LeftAuthority290207.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290531

namespace LeftBound290535
def owner : Owner := ⟨.program ⟨257⟩, ⟨66186⟩⟩
def transferEvent : Nat := 290535
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290533 .coefficient, .predecessor 1 290534 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290533 .coefficient)
      LeftBound290531.bound (LeftBound290531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290532RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290531.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290534 .coefficient)
      LeftAuthority290184.bound (LeftAuthority290184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290531.bound, LeftAuthority290184.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290531.bound, LeftAuthority290184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290531.actual selector witness, LeftAuthority290184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290535

namespace LeftBound290539
def owner : Owner := ⟨.program ⟨257⟩, ⟨66187⟩⟩
def transferEvent : Nat := 290539
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290537 .coefficient, .predecessor 1 290538 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290537 .coefficient)
      LeftBound290535.bound (LeftBound290535.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290536RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290535.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290535.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290538 .coefficient)
      LeftAuthority290161.bound (LeftAuthority290161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290535.bound, LeftAuthority290161.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290535.bound, LeftAuthority290161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290535.actual selector witness, LeftAuthority290161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290539

namespace LeftBound290543
def owner : Owner := ⟨.program ⟨257⟩, ⟨66188⟩⟩
def transferEvent : Nat := 290543
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290541 .coefficient, .predecessor 1 290542 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290541 .coefficient)
      LeftBound290539.bound (LeftBound290539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290542 .coefficient)
      LeftAuthority290138.bound (LeftAuthority290138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290539.bound, LeftAuthority290138.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290539.bound, LeftAuthority290138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290539.actual selector witness, LeftAuthority290138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290543

namespace LeftBound290547
def owner : Owner := ⟨.program ⟨257⟩, ⟨66189⟩⟩
def transferEvent : Nat := 290547
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290545 .coefficient, .predecessor 1 290546 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290545 .coefficient)
      LeftBound290543.bound (LeftBound290543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290544RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290546 .coefficient)
      LeftAuthority290115.bound (LeftAuthority290115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290115.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290543.bound, LeftAuthority290115.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290543.bound, LeftAuthority290115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290543.actual selector witness, LeftAuthority290115.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290547

namespace LeftBound290551
def owner : Owner := ⟨.program ⟨257⟩, ⟨66190⟩⟩
def transferEvent : Nat := 290551
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290549 .coefficient, .predecessor 1 290550 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290549 .coefficient)
      LeftBound290547.bound (LeftBound290547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290547.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290550 .coefficient)
      LeftAuthority290092.bound (LeftAuthority290092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1133.exact290093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290092.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290547.bound, LeftAuthority290092.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290547.bound, LeftAuthority290092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290547.actual selector witness, LeftAuthority290092.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290551

namespace LeftBound290554
def owner : Owner := ⟨.program ⟨257⟩, ⟨66191⟩⟩
def transferEvent : Nat := 290554
def frameStart : Nat := 290050
def rule : BoundRule := .identity (.predecessor 0 290553 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290553 .coefficient)
      LeftBound290551.bound (LeftBound290551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1134.exact290552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290551.derived selector witness)

def rawBound : CoeffClass := LeftBound290551.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound290551.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound290554

namespace LeftBound290571
def owner : Owner := ⟨.program ⟨257⟩, ⟨69063⟩⟩
def transferEvent : Nat := 290571
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290569 .coefficient, .predecessor 1 290570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290569 .coefficient)
      LeftBound290554.bound (LeftBound290554.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound290554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290570 .coefficient)
      LeftAuthority290567.bound (LeftAuthority290567.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority290567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290554.bound, LeftAuthority290567.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290554.bound, LeftAuthority290567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290554.actual selector witness, LeftAuthority290567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290571

namespace LeftBound290574
def owner : Owner := ⟨.program ⟨257⟩, ⟨69064⟩⟩
def transferEvent : Nat := 290574
def frameStart : Nat := 290050
def rule : BoundRule := .identity (.predecessor 0 290573 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290573 .coefficient)
      LeftBound290571.bound (LeftBound290571.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound290571.derived selector witness)

def rawBound : CoeffClass := LeftBound290571.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290571.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound290571.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound290574

namespace LeftBound290580
def owner : Owner := ⟨.program ⟨257⟩, ⟨69065⟩⟩
def transferEvent : Nat := 290580
def frameStart : Nat := 290050
def rule : BoundRule := .product (.predecessor 0 290578 .coefficient) (.predecessor 1 290579 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290578 .coefficient)
      LeftAuthority290576.bound (LeftAuthority290576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290579 .coefficient)
      LeftBound290574.bound (LeftBound290574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority290576.bound LeftBound290574.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority290576.bound, LeftBound290574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority290576.actual selector witness) * (LeftBound290574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound290580

namespace LeftBound290656
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 290656
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290654 .coefficient, .predecessor 1 290655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290654 .coefficient)
      LeftAuthority290652.bound (LeftAuthority290652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290655 .coefficient)
      LeftAuthority290649.bound (LeftAuthority290649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority290652.bound, LeftAuthority290649.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority290652.bound, LeftAuthority290649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority290652.actual selector witness, LeftAuthority290649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290656

namespace LeftBound290660
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 290660
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290658 .coefficient, .predecessor 1 290659 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290658 .coefficient)
      LeftBound290656.bound (LeftBound290656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290659 .coefficient)
      LeftAuthority290646.bound (LeftAuthority290646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290656.bound, LeftAuthority290646.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290656.bound, LeftAuthority290646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290656.actual selector witness, LeftAuthority290646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290660

namespace LeftBound290664
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 290664
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290662 .coefficient, .predecessor 1 290663 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290662 .coefficient)
      LeftBound290660.bound (LeftBound290660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290663 .coefficient)
      LeftAuthority290643.bound (LeftAuthority290643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290660.bound, LeftAuthority290643.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290660.bound, LeftAuthority290643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290660.actual selector witness, LeftAuthority290643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290664

namespace LeftBound290668
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 290668
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290666 .coefficient, .predecessor 1 290667 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290666 .coefficient)
      LeftBound290664.bound (LeftBound290664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290664.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290667 .coefficient)
      LeftAuthority290640.bound (LeftAuthority290640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290640.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290664.bound, LeftAuthority290640.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290664.bound, LeftAuthority290640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290664.actual selector witness, LeftAuthority290640.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290668

namespace LeftBound290672
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 290672
def frameStart : Nat := 290050
def rule : BoundRule := .sum [.predecessor 0 290670 .coefficient, .predecessor 1 290671 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 290670 .coefficient)
      LeftBound290668.bound (LeftBound290668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290669RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290668.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 290671 .coefficient)
      LeftAuthority290637.bound (LeftAuthority290637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1135.exact290638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority290637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority290637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound290668.bound, LeftAuthority290637.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound290668.bound, LeftAuthority290637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound290668.actual selector witness, LeftAuthority290637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound290672

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
