import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1054

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound159083
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 159083
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159081 .coefficient, .predecessor 1 159082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159081 .coefficient)
      LeftBound159079.bound (LeftBound159079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159082 .coefficient)
      LeftAuthority159048.bound (LeftAuthority159048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159079.bound, LeftAuthority159048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159079.bound, LeftAuthority159048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159079.actual selector witness, LeftAuthority159048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159083

namespace LeftBound159087
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 159087
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159085 .coefficient, .predecessor 1 159086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159085 .coefficient)
      LeftBound159083.bound (LeftBound159083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159086 .coefficient)
      LeftAuthority159045.bound (LeftAuthority159045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159045.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159045.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159083.bound, LeftAuthority159045.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159083.bound, LeftAuthority159045.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159083.actual selector witness, LeftAuthority159045.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159087

namespace LeftBound159091
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 159091
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159089 .coefficient, .predecessor 1 159090 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159089 .coefficient)
      LeftBound159087.bound (LeftBound159087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159090 .coefficient)
      LeftAuthority159042.bound (LeftAuthority159042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159087.bound, LeftAuthority159042.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159087.bound, LeftAuthority159042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159087.actual selector witness, LeftAuthority159042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159091

namespace LeftBound159095
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 159095
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159093 .coefficient, .predecessor 1 159094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159093 .coefficient)
      LeftBound159091.bound (LeftBound159091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159094 .coefficient)
      LeftAuthority159039.bound (LeftAuthority159039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159091.bound, LeftAuthority159039.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159091.bound, LeftAuthority159039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159091.actual selector witness, LeftAuthority159039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159095

namespace LeftBound159099
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 159099
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159097 .coefficient, .predecessor 1 159098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159097 .coefficient)
      LeftBound159095.bound (LeftBound159095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159098 .coefficient)
      LeftAuthority159036.bound (LeftAuthority159036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159095.bound, LeftAuthority159036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159095.bound, LeftAuthority159036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159095.actual selector witness, LeftAuthority159036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159099

namespace LeftBound159103
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 159103
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159101 .coefficient, .predecessor 1 159102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159101 .coefficient)
      LeftBound159099.bound (LeftBound159099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159102 .coefficient)
      LeftAuthority159033.bound (LeftAuthority159033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159033.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159099.bound, LeftAuthority159033.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159099.bound, LeftAuthority159033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159099.actual selector witness, LeftAuthority159033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159103

namespace LeftBound159107
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 159107
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159105 .coefficient, .predecessor 1 159106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159105 .coefficient)
      LeftBound159103.bound (LeftBound159103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159106 .coefficient)
      LeftAuthority159030.bound (LeftAuthority159030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159103.bound, LeftAuthority159030.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159103.bound, LeftAuthority159030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159103.actual selector witness, LeftAuthority159030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159107

namespace LeftBound159111
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 159111
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159109 .coefficient, .predecessor 1 159110 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159109 .coefficient)
      LeftBound159107.bound (LeftBound159107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159110 .coefficient)
      LeftAuthority159027.bound (LeftAuthority159027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159027.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159107.bound, LeftAuthority159027.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159107.bound, LeftAuthority159027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159107.actual selector witness, LeftAuthority159027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159111

namespace LeftBound159115
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 159115
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159113 .coefficient, .predecessor 1 159114 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159113 .coefficient)
      LeftBound159111.bound (LeftBound159111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159114 .coefficient)
      LeftAuthority159024.bound (LeftAuthority159024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159024.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159024.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159111.bound, LeftAuthority159024.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159111.bound, LeftAuthority159024.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159111.actual selector witness, LeftAuthority159024.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159115

namespace LeftBound159119
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 159119
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159117 .coefficient, .predecessor 1 159118 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159117 .coefficient)
      LeftBound159115.bound (LeftBound159115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159118 .coefficient)
      LeftAuthority159021.bound (LeftAuthority159021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159115.bound, LeftAuthority159021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159115.bound, LeftAuthority159021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159115.actual selector witness, LeftAuthority159021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159119

namespace LeftBound159123
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 159123
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159121 .coefficient, .predecessor 1 159122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159121 .coefficient)
      LeftBound159119.bound (LeftBound159119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159122 .coefficient)
      LeftAuthority159018.bound (LeftAuthority159018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159018.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159018.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159119.bound, LeftAuthority159018.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159119.bound, LeftAuthority159018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159119.actual selector witness, LeftAuthority159018.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159123

namespace LeftBound159127
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 159127
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159125 .coefficient, .predecessor 1 159126 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159125 .coefficient)
      LeftBound159123.bound (LeftBound159123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159126 .coefficient)
      LeftAuthority159015.bound (LeftAuthority159015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159015.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159123.bound, LeftAuthority159015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159123.bound, LeftAuthority159015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159123.actual selector witness, LeftAuthority159015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159127

namespace LeftBound159131
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 159131
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159129 .coefficient, .predecessor 1 159130 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159129 .coefficient)
      LeftBound159127.bound (LeftBound159127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159127.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159130 .coefficient)
      LeftAuthority159012.bound (LeftAuthority159012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159012.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159127.bound, LeftAuthority159012.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159127.bound, LeftAuthority159012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159127.actual selector witness, LeftAuthority159012.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159131

namespace LeftBound159135
def owner : Owner := ⟨.program ⟨257⟩, ⟨69078⟩⟩
def transferEvent : Nat := 159135
def frameStart : Nat := 158461
def rule : BoundRule := .sum [.predecessor 0 159133 .coefficient, .predecessor 1 159134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159133 .coefficient)
      LeftBound159131.bound (LeftBound159131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159134 .coefficient)
      LeftBound158991.bound (LeftBound158991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound158991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound158991.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159131.bound, LeftBound158991.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159131.bound, LeftBound158991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159131.actual selector witness, LeftBound158991.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound159135

namespace LeftBound159139
def owner : Owner := ⟨.program ⟨257⟩, ⟨71143⟩⟩
def transferEvent : Nat := 159139
def frameStart : Nat := 158461
def rule : BoundRule := .product (.predecessor 0 159137 .coefficient) (.predecessor 1 159138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159137 .coefficient)
      LeftBound159135.bound (LeftBound159135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159138 .coefficient)
      LeftAuthority158976.bound (LeftAuthority158976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact158977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158976.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound159135.bound LeftAuthority158976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159135.bound, LeftAuthority158976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound159135.actual selector witness) * (LeftAuthority158976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound159139

namespace LeftBound159218
def owner : Owner := ⟨.program ⟨257⟩, ⟨67401⟩⟩
def transferEvent : Nat := 159218
def frameStart : Nat := 158461
def rule : BoundRule := .product (.predecessor 0 159216 .coefficient) (.predecessor 1 159217 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 159216 .coefficient)
      LeftAuthority158987.bound (LeftAuthority158987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact158988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority158987.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority158987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 159217 .coefficient)
      LeftAuthority159214.bound (LeftAuthority159214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events621.exact159215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority159214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority159214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority158987.bound LeftAuthority159214.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority158987.bound, LeftAuthority159214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority158987.actual selector witness) * (LeftAuthority159214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound159218

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
