import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard304
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard308
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard311
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard315
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard318
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard319
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard322
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard326
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard329
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard333
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard340

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55329
def owner : Owner := ⟨.program ⟨257⟩, ⟨24124⟩⟩
def transferEvent : Nat := 55329
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55325 .summary, .result 54356 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55325 .summary)
      LeftBound55324.bound (LeftBound55324.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20904⟩⟩) (rawTerms := some (Proof.Events216.exact55325RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55324.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54356 .summary)
      LeftBound54355.bound (LeftBound54355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24123⟩⟩) (rawTerms := some (Proof.Events212.exact54356RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54355.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55324.bound, LeftBound54355.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55324.bound, LeftBound54355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55324.actual selector witness, LeftBound54355.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55329

namespace LeftBound55333
def owner : Owner := ⟨.program ⟨257⟩, ⟨34144⟩⟩
def transferEvent : Nat := 55333
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55331 .coefficient, .predecessor 1 55332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55331 .coefficient)
      LeftBound55328.bound (LeftBound55328.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55328.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55332 .coefficient)
      LeftBound53870.bound (LeftBound53870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53870.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55328.bound, LeftBound53870.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55328.bound, LeftBound53870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55328.actual selector witness, LeftBound53870.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55333

namespace LeftBound55334
def owner : Owner := ⟨.program ⟨257⟩, ⟨34144⟩⟩
def transferEvent : Nat := 55334
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55330 .summary, .result 53874 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55330 .summary)
      LeftBound55329.bound (LeftBound55329.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24124⟩⟩) (rawTerms := some (Proof.Events216.exact55330RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53874 .summary)
      LeftBound53873.bound (LeftBound53873.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34143⟩⟩) (rawTerms := some (Proof.Events210.exact53874RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55329.bound, LeftBound53873.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55329.bound, LeftBound53873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55329.actual selector witness, LeftBound53873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55334

namespace LeftBound55338
def owner : Owner := ⟨.program ⟨257⟩, ⟨53204⟩⟩
def transferEvent : Nat := 55338
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55336 .coefficient, .predecessor 1 55337 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55336 .coefficient)
      LeftBound55333.bound (LeftBound55333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55337 .coefficient)
      LeftBound53388.bound (LeftBound53388.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53392RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53388.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55333.bound, LeftBound53388.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55333.bound, LeftBound53388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55333.actual selector witness, LeftBound53388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55338

namespace LeftBound55339
def owner : Owner := ⟨.program ⟨257⟩, ⟨53204⟩⟩
def transferEvent : Nat := 55339
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55335 .summary, .result 53392 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55335 .summary)
      LeftBound55334.bound (LeftBound55334.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34144⟩⟩) (rawTerms := some (Proof.Events216.exact55335RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53392 .summary)
      LeftBound53391.bound (LeftBound53391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53203⟩⟩) (rawTerms := some (Proof.Events208.exact53392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55334.bound, LeftBound53391.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55334.bound, LeftBound53391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55334.actual selector witness, LeftBound53391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55339

namespace LeftBound55343
def owner : Owner := ⟨.program ⟨257⟩, ⟨56184⟩⟩
def transferEvent : Nat := 55343
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55341 .coefficient, .predecessor 1 55342 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55341 .coefficient)
      LeftBound55338.bound (LeftBound55338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55342 .coefficient)
      LeftBound52906.bound (LeftBound52906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events206.exact52910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55338.bound, LeftBound52906.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55338.bound, LeftBound52906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55338.actual selector witness, LeftBound52906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55343

namespace LeftBound55344
def owner : Owner := ⟨.program ⟨257⟩, ⟨56184⟩⟩
def transferEvent : Nat := 55344
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55340 .summary, .result 52910 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55340 .summary)
      LeftBound55339.bound (LeftBound55339.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53204⟩⟩) (rawTerms := some (Proof.Events216.exact55340RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 52910 .summary)
      LeftBound52909.bound (LeftBound52909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56183⟩⟩) (rawTerms := some (Proof.Events206.exact52910RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55339.bound, LeftBound52909.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55339.bound, LeftBound52909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55339.actual selector witness, LeftBound52909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55344

namespace LeftBound55348
def owner : Owner := ⟨.program ⟨257⟩, ⟨59164⟩⟩
def transferEvent : Nat := 55348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55346 .coefficient, .predecessor 1 55347 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55346 .coefficient)
      LeftBound55343.bound (LeftBound55343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55347 .coefficient)
      LeftBound52424.bound (LeftBound52424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55343.bound, LeftBound52424.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55343.bound, LeftBound52424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55343.actual selector witness, LeftBound52424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55348

namespace LeftBound55349
def owner : Owner := ⟨.program ⟨257⟩, ⟨59164⟩⟩
def transferEvent : Nat := 55349
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55345 .summary, .result 52428 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55345 .summary)
      LeftBound55344.bound (LeftBound55344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56184⟩⟩) (rawTerms := some (Proof.Events216.exact55345RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 52428 .summary)
      LeftBound52427.bound (LeftBound52427.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59163⟩⟩) (rawTerms := some (Proof.Events204.exact52428RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52427.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55344.bound, LeftBound52427.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55344.bound, LeftBound52427.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55344.actual selector witness, LeftBound52427.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55349

namespace LeftBound55353
def owner : Owner := ⟨.program ⟨257⟩, ⟨62144⟩⟩
def transferEvent : Nat := 55353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55351 .coefficient, .predecessor 1 55352 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55351 .coefficient)
      LeftBound55348.bound (LeftBound55348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55348.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55352 .coefficient)
      LeftBound51942.bound (LeftBound51942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51942.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55348.bound, LeftBound51942.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55348.bound, LeftBound51942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55348.actual selector witness, LeftBound51942.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55353

namespace LeftBound55354
def owner : Owner := ⟨.program ⟨257⟩, ⟨62144⟩⟩
def transferEvent : Nat := 55354
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55350 .summary, .result 51946 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55350 .summary)
      LeftBound55349.bound (LeftBound55349.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59164⟩⟩) (rawTerms := some (Proof.Events216.exact55350RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55349.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 51946 .summary)
      LeftBound51945.bound (LeftBound51945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62143⟩⟩) (rawTerms := some (Proof.Events202.exact51946RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55349.bound, LeftBound51945.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55349.bound, LeftBound51945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55349.actual selector witness, LeftBound51945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55354

namespace LeftBound55358
def owner : Owner := ⟨.program ⟨257⟩, ⟨65124⟩⟩
def transferEvent : Nat := 55358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55356 .coefficient, .predecessor 1 55357 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55356 .coefficient)
      LeftBound55353.bound (LeftBound55353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55355RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55357 .coefficient)
      LeftBound51460.bound (LeftBound51460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events201.exact51464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55353.bound, LeftBound51460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55353.bound, LeftBound51460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55353.actual selector witness, LeftBound51460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55358

namespace LeftBound55359
def owner : Owner := ⟨.program ⟨257⟩, ⟨65124⟩⟩
def transferEvent : Nat := 55359
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55355 .summary, .result 51464 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55355 .summary)
      LeftBound55354.bound (LeftBound55354.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62144⟩⟩) (rawTerms := some (Proof.Events216.exact55355RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55354.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 51464 .summary)
      LeftBound51463.bound (LeftBound51463.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65123⟩⟩) (rawTerms := some (Proof.Events201.exact51464RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound51463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55354.bound, LeftBound51463.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55354.bound, LeftBound51463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55354.actual selector witness, LeftBound51463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55359

namespace LeftBound55363
def owner : Owner := ⟨.program ⟨257⟩, ⟨70813⟩⟩
def transferEvent : Nat := 55363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55361 .coefficient, .predecessor 1 55362 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55361 .coefficient)
      LeftBound55358.bound (LeftBound55358.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55360RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55358.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55362 .coefficient)
      LeftBound50978.bound (LeftBound50978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55358.bound, LeftBound50978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55358.bound, LeftBound50978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55358.actual selector witness, LeftBound50978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55363

namespace LeftBound55364
def owner : Owner := ⟨.program ⟨257⟩, ⟨70813⟩⟩
def transferEvent : Nat := 55364
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55360 .summary, .result 50982 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 55360 .summary)
      LeftBound55359.bound (LeftBound55359.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65124⟩⟩) (rawTerms := some (Proof.Events216.exact55360RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55359.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 50982 .summary)
      LeftBound50981.bound (LeftBound50981.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70812⟩⟩) (rawTerms := some (Proof.Events199.exact50982RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50981.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55359.bound, LeftBound50981.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55359.bound, LeftBound50981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55359.actual selector witness, LeftBound50981.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55364

namespace LeftBound55368
def owner : Owner := ⟨.program ⟨257⟩, ⟨70814⟩⟩
def transferEvent : Nat := 55368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55366 .coefficient, .predecessor 1 55367 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 55366 .coefficient)
      LeftBound55363.bound (LeftBound55363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55365RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55363.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 55367 .coefficient)
      LeftBound50496.bound (LeftBound50496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50500RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50496.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55363.bound, LeftBound50496.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55363.bound, LeftBound50496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound55363.actual selector witness, LeftBound50496.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55368

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
