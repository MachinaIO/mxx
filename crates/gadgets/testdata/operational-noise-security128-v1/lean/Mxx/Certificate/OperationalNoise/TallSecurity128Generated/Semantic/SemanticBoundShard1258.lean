import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1257

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound188341
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 188341
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188339 .coefficient, .predecessor 1 188340 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188339 .coefficient)
      LeftBound188337.bound (LeftBound188337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188340 .coefficient)
      LeftAuthority188292.bound (LeftAuthority188292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188293RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188292.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188292.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188337.bound, LeftAuthority188292.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188337.bound, LeftAuthority188292.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188337.actual selector witness, LeftAuthority188292.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188341

namespace LeftBound188345
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 188345
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188343 .coefficient, .predecessor 1 188344 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188343 .coefficient)
      LeftBound188341.bound (LeftBound188341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188342RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188344 .coefficient)
      LeftAuthority188289.bound (LeftAuthority188289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188289.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188289.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188341.bound, LeftAuthority188289.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188341.bound, LeftAuthority188289.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188341.actual selector witness, LeftAuthority188289.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188345

namespace LeftBound188349
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 188349
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188347 .coefficient, .predecessor 1 188348 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188347 .coefficient)
      LeftBound188345.bound (LeftBound188345.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188346RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188345.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188345.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188348 .coefficient)
      LeftAuthority188286.bound (LeftAuthority188286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188286.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188345.bound, LeftAuthority188286.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188345.bound, LeftAuthority188286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188345.actual selector witness, LeftAuthority188286.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188349

namespace LeftBound188353
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 188353
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188351 .coefficient, .predecessor 1 188352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188351 .coefficient)
      LeftBound188349.bound (LeftBound188349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188349.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188352 .coefficient)
      LeftAuthority188283.bound (LeftAuthority188283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188283.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188283.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188349.bound, LeftAuthority188283.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188349.bound, LeftAuthority188283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188349.actual selector witness, LeftAuthority188283.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188353

namespace LeftBound188357
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 188357
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188355 .coefficient, .predecessor 1 188356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188355 .coefficient)
      LeftBound188353.bound (LeftBound188353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188356 .coefficient)
      LeftAuthority188280.bound (LeftAuthority188280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188280.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188353.bound, LeftAuthority188280.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188353.bound, LeftAuthority188280.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188353.actual selector witness, LeftAuthority188280.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188357

namespace LeftBound188361
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 188361
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188359 .coefficient, .predecessor 1 188360 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188359 .coefficient)
      LeftBound188357.bound (LeftBound188357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188360 .coefficient)
      LeftAuthority188277.bound (LeftAuthority188277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188277.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188277.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188357.bound, LeftAuthority188277.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188357.bound, LeftAuthority188277.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188357.actual selector witness, LeftAuthority188277.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188361

namespace LeftBound188365
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 188365
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188363 .coefficient, .predecessor 1 188364 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188363 .coefficient)
      LeftBound188361.bound (LeftBound188361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188364 .coefficient)
      LeftAuthority188274.bound (LeftAuthority188274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188274.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188361.bound, LeftAuthority188274.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188361.bound, LeftAuthority188274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188361.actual selector witness, LeftAuthority188274.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188365

namespace LeftBound188369
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 188369
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188367 .coefficient, .predecessor 1 188368 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188367 .coefficient)
      LeftBound188365.bound (LeftBound188365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188368 .coefficient)
      LeftAuthority188271.bound (LeftAuthority188271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188271.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188271.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188365.bound, LeftAuthority188271.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188365.bound, LeftAuthority188271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188365.actual selector witness, LeftAuthority188271.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188369

namespace LeftBound188373
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 188373
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188371 .coefficient, .predecessor 1 188372 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188371 .coefficient)
      LeftBound188369.bound (LeftBound188369.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188369.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188369.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188372 .coefficient)
      LeftAuthority188268.bound (LeftAuthority188268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188268.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188268.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188369.bound, LeftAuthority188268.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188369.bound, LeftAuthority188268.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188369.actual selector witness, LeftAuthority188268.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188373

namespace LeftBound188377
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 188377
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188375 .coefficient, .predecessor 1 188376 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188375 .coefficient)
      LeftBound188373.bound (LeftBound188373.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188374RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188373.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188373.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188376 .coefficient)
      LeftAuthority188265.bound (LeftAuthority188265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188265.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188373.bound, LeftAuthority188265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188373.bound, LeftAuthority188265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188373.actual selector witness, LeftAuthority188265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188377

namespace LeftBound188381
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 188381
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188379 .coefficient, .predecessor 1 188380 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188379 .coefficient)
      LeftBound188377.bound (LeftBound188377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188377.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188377.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188380 .coefficient)
      LeftAuthority188262.bound (LeftAuthority188262.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188263RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188262.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188377.bound, LeftAuthority188262.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188377.bound, LeftAuthority188262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188377.actual selector witness, LeftAuthority188262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188381

namespace LeftBound188385
def owner : Owner := ⟨.program ⟨257⟩, ⟨69102⟩⟩
def transferEvent : Nat := 188385
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188383 .coefficient, .predecessor 1 188384 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188383 .coefficient)
      LeftBound188381.bound (LeftBound188381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188381.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188384 .coefficient)
      LeftBound188241.bound (LeftBound188241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188241.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188381.bound, LeftBound188241.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188381.bound, LeftBound188241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188381.actual selector witness, LeftBound188241.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188385

namespace LeftBound188389
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def transferEvent : Nat := 188389
def frameStart : Nat := 187711
def rule : BoundRule := .product (.predecessor 0 188387 .coefficient) (.predecessor 1 188388 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188387 .coefficient)
      LeftBound188385.bound (LeftBound188385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188388 .coefficient)
      LeftAuthority188226.bound (LeftAuthority188226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188226.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound188385.bound LeftAuthority188226.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188385.bound, LeftAuthority188226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound188385.actual selector witness) * (LeftAuthority188226.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound188389

namespace LeftBound188468
def owner : Owner := ⟨.program ⟨257⟩, ⟨67516⟩⟩
def transferEvent : Nat := 188468
def frameStart : Nat := 187711
def rule : BoundRule := .product (.predecessor 0 188466 .coefficient) (.predecessor 1 188467 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188466 .coefficient)
      LeftAuthority188237.bound (LeftAuthority188237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188237.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188467 .coefficient)
      LeftAuthority188464.bound (LeftAuthority188464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events736.exact188465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188464.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority188237.bound LeftAuthority188464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188237.bound, LeftAuthority188464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority188237.actual selector witness) * (LeftAuthority188464.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound188468

namespace LeftBound188476
def owner : Owner := ⟨.program ⟨257⟩, ⟨67521⟩⟩
def transferEvent : Nat := 188476
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188474 .coefficient, .predecessor 1 188475 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188474 .coefficient)
      LeftAuthority188472.bound (LeftAuthority188472.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events736.exact188473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188472.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188472.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188475 .coefficient)
      LeftBound188468.bound (LeftBound188468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events736.exact188470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188468.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188468.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority188472.bound, LeftBound188468.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188472.bound, LeftBound188468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority188472.actual selector witness, LeftBound188468.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188476

namespace LeftBound188480
def owner : Owner := ⟨.program ⟨257⟩, ⟨71334⟩⟩
def transferEvent : Nat := 188480
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188478 .coefficient, .predecessor 1 188479 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188478 .coefficient)
      LeftBound188476.bound (LeftBound188476.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events736.exact188477RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188476.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188476.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188479 .coefficient)
      LeftBound188389.bound (LeftBound188389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events736.exact188462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188476.bound, LeftBound188389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188476.bound, LeftBound188389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188476.actual selector witness, LeftBound188389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188480

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
