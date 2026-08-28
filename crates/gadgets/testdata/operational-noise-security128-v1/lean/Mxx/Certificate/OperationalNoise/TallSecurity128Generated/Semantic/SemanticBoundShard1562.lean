import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1561

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound232196
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 232196
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232194 .coefficient, .predecessor 1 232195 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232194 .coefficient)
      LeftBound232192.bound (LeftBound232192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232195 .coefficient)
      LeftAuthority232182.bound (LeftAuthority232182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232192.bound, LeftAuthority232182.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232192.bound, LeftAuthority232182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232192.actual selector witness, LeftAuthority232182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232196

namespace LeftBound232200
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 232200
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232198 .coefficient, .predecessor 1 232199 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232198 .coefficient)
      LeftBound232196.bound (LeftBound232196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232196.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232199 .coefficient)
      LeftAuthority232179.bound (LeftAuthority232179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232196.bound, LeftAuthority232179.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232196.bound, LeftAuthority232179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232196.actual selector witness, LeftAuthority232179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232200

namespace LeftBound232204
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 232204
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232202 .coefficient, .predecessor 1 232203 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232202 .coefficient)
      LeftBound232200.bound (LeftBound232200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232200.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232200.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232203 .coefficient)
      LeftAuthority232176.bound (LeftAuthority232176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232200.bound, LeftAuthority232176.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232200.bound, LeftAuthority232176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232200.actual selector witness, LeftAuthority232176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232204

namespace LeftBound232208
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 232208
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232206 .coefficient, .predecessor 1 232207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232206 .coefficient)
      LeftBound232204.bound (LeftBound232204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232207 .coefficient)
      LeftAuthority232173.bound (LeftAuthority232173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232173.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232204.bound, LeftAuthority232173.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232204.bound, LeftAuthority232173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232204.actual selector witness, LeftAuthority232173.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232208

namespace LeftBound232212
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 232212
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232210 .coefficient, .predecessor 1 232211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232210 .coefficient)
      LeftBound232208.bound (LeftBound232208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232211 .coefficient)
      LeftAuthority232170.bound (LeftAuthority232170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232170.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232208.bound, LeftAuthority232170.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232208.bound, LeftAuthority232170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232208.actual selector witness, LeftAuthority232170.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232212

namespace LeftBound232216
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 232216
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232214 .coefficient, .predecessor 1 232215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232214 .coefficient)
      LeftBound232212.bound (LeftBound232212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232215 .coefficient)
      LeftAuthority232167.bound (LeftAuthority232167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232212.bound, LeftAuthority232167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232212.bound, LeftAuthority232167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232212.actual selector witness, LeftAuthority232167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232216

namespace LeftBound232220
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 232220
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232218 .coefficient, .predecessor 1 232219 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232218 .coefficient)
      LeftBound232216.bound (LeftBound232216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232219 .coefficient)
      LeftAuthority232164.bound (LeftAuthority232164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232164.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232164.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232216.bound, LeftAuthority232164.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232216.bound, LeftAuthority232164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232216.actual selector witness, LeftAuthority232164.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232220

namespace LeftBound232224
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 232224
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232222 .coefficient, .predecessor 1 232223 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232222 .coefficient)
      LeftBound232220.bound (LeftBound232220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232220.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232220.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232223 .coefficient)
      LeftAuthority232161.bound (LeftAuthority232161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232162RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232161.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232220.bound, LeftAuthority232161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232220.bound, LeftAuthority232161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232220.actual selector witness, LeftAuthority232161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232224

namespace LeftBound232228
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 232228
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232226 .coefficient, .predecessor 1 232227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232226 .coefficient)
      LeftBound232224.bound (LeftBound232224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232227 .coefficient)
      LeftAuthority232158.bound (LeftAuthority232158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232224.bound, LeftAuthority232158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232224.bound, LeftAuthority232158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232224.actual selector witness, LeftAuthority232158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232228

namespace LeftBound232232
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 232232
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232230 .coefficient, .predecessor 1 232231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232230 .coefficient)
      LeftBound232228.bound (LeftBound232228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232231 .coefficient)
      LeftAuthority232155.bound (LeftAuthority232155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232228.bound, LeftAuthority232155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232228.bound, LeftAuthority232155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232228.actual selector witness, LeftAuthority232155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232232

namespace LeftBound232236
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 232236
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232234 .coefficient, .predecessor 1 232235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232234 .coefficient)
      LeftBound232232.bound (LeftBound232232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232235 .coefficient)
      LeftAuthority232152.bound (LeftAuthority232152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232232.bound, LeftAuthority232152.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232232.bound, LeftAuthority232152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232232.actual selector witness, LeftAuthority232152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232236

namespace LeftBound232240
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 232240
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232238 .coefficient, .predecessor 1 232239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232238 .coefficient)
      LeftBound232236.bound (LeftBound232236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232239 .coefficient)
      LeftAuthority232149.bound (LeftAuthority232149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232236.bound, LeftAuthority232149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232236.bound, LeftAuthority232149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232236.actual selector witness, LeftAuthority232149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232240

namespace LeftBound232244
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 232244
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232242 .coefficient, .predecessor 1 232243 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232242 .coefficient)
      LeftBound232240.bound (LeftBound232240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232243 .coefficient)
      LeftAuthority232146.bound (LeftAuthority232146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232146.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232240.bound, LeftAuthority232146.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232240.bound, LeftAuthority232146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232240.actual selector witness, LeftAuthority232146.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232244

namespace LeftBound232248
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 232248
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232246 .coefficient, .predecessor 1 232247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232246 .coefficient)
      LeftBound232244.bound (LeftBound232244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232247 .coefficient)
      LeftAuthority232143.bound (LeftAuthority232143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232143.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232244.bound, LeftAuthority232143.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232244.bound, LeftAuthority232143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232244.actual selector witness, LeftAuthority232143.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232248

namespace LeftBound232252
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 232252
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232250 .coefficient, .predecessor 1 232251 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232250 .coefficient)
      LeftBound232248.bound (LeftBound232248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232251 .coefficient)
      LeftAuthority232140.bound (LeftAuthority232140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232248.bound, LeftAuthority232140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232248.bound, LeftAuthority232140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232248.actual selector witness, LeftAuthority232140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232252

namespace LeftBound232256
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 232256
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232254 .coefficient, .predecessor 1 232255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232254 .coefficient)
      LeftBound232252.bound (LeftBound232252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events907.exact232253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232255 .coefficient)
      LeftAuthority232137.bound (LeftAuthority232137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232137.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232252.bound, LeftAuthority232137.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232252.bound, LeftAuthority232137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232252.actual selector witness, LeftAuthority232137.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232256

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
