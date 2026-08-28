import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1286
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1289
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1321

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound197174
def owner : Owner := ⟨.program ⟨257⟩, ⟨69016⟩⟩
def transferEvent : Nat := 197174
def frameStart : Nat := 197115
def rule : BoundRule := .identity (.predecessor 0 197173 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197173 .coefficient)
      LeftBound197171.bound (LeftBound197171.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound197171.derived selector witness)

def rawBound : CoeffClass := LeftBound197171.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound197171.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound197174

namespace LeftBound197180
def owner : Owner := ⟨.program ⟨257⟩, ⟨69017⟩⟩
def transferEvent : Nat := 197180
def frameStart : Nat := 197115
def rule : BoundRule := .product (.predecessor 0 197178 .coefficient) (.predecessor 1 197179 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197178 .coefficient)
      LeftAuthority197176.bound (LeftAuthority197176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197179 .coefficient)
      LeftBound197174.bound (LeftBound197174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197174.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority197176.bound LeftBound197174.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority197176.bound, LeftBound197174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority197176.actual selector witness) * (LeftBound197174.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound197180

namespace LeftBound197188
def owner : Owner := ⟨.program ⟨257⟩, ⟨69018⟩⟩
def transferEvent : Nat := 197188
def frameStart : Nat := 197115
def rule : BoundRule := .sum [.predecessor 0 197186 .coefficient, .predecessor 1 197187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197186 .coefficient)
      LeftAuthority197184.bound (LeftAuthority197184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197187 .coefficient)
      LeftBound197180.bound (LeftBound197180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197182RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority197184.bound, LeftBound197180.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority197184.bound, LeftBound197180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority197184.actual selector witness, LeftBound197180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197188

namespace LeftBound197192
def owner : Owner := ⟨.program ⟨257⟩, ⟨70336⟩⟩
def transferEvent : Nat := 197192
def frameStart : Nat := 197115
def rule : BoundRule := .product (.predecessor 0 197190 .coefficient) (.predecessor 1 197191 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197190 .coefficient)
      LeftBound197188.bound (LeftBound197188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197191 .coefficient)
      LeftAuthority197165.bound (LeftAuthority197165.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197165.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound197188.bound LeftAuthority197165.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197188.bound, LeftAuthority197165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound197188.actual selector witness) * (LeftAuthority197165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound197192

namespace LeftBound197203
def owner : Owner := ⟨.program ⟨257⟩, ⟨66752⟩⟩
def transferEvent : Nat := 197203
def frameStart : Nat := 197115
def rule : BoundRule := .product (.predecessor 0 197201 .coefficient) (.predecessor 1 197202 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197201 .coefficient)
      LeftAuthority197176.bound (LeftAuthority197176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197176.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197202 .coefficient)
      LeftAuthority197199.bound (LeftAuthority197199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority197176.bound LeftAuthority197199.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority197176.bound, LeftAuthority197199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority197176.actual selector witness) * (LeftAuthority197199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound197203

namespace LeftBound197211
def owner : Owner := ⟨.program ⟨257⟩, ⟨66753⟩⟩
def transferEvent : Nat := 197211
def frameStart : Nat := 197115
def rule : BoundRule := .sum [.predecessor 0 197209 .coefficient, .predecessor 1 197210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197209 .coefficient)
      LeftAuthority197207.bound (LeftAuthority197207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority197207.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority197207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197210 .coefficient)
      LeftBound197203.bound (LeftBound197203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197203.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority197207.bound, LeftBound197203.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority197207.bound, LeftBound197203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority197207.actual selector witness, LeftBound197203.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197211

namespace LeftBound197215
def owner : Owner := ⟨.program ⟨257⟩, ⟨70348⟩⟩
def transferEvent : Nat := 197215
def frameStart : Nat := 197115
def rule : BoundRule := .sum [.predecessor 0 197213 .coefficient, .predecessor 1 197214 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197213 .coefficient)
      LeftBound197211.bound (LeftBound197211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197214 .coefficient)
      LeftBound197192.bound (LeftBound197192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197192.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197192.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound197211.bound, LeftBound197192.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197211.bound, LeftBound197192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound197211.actual selector witness, LeftBound197192.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197215

namespace LeftBound197228
def owner : Owner := ⟨.program ⟨257⟩, ⟨70338⟩⟩
def transferEvent : Nat := 197228
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 197226 .coefficient, .predecessor 1 197227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197226 .coefficient)
      LeftBound197057.bound (LeftBound197057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197227 .coefficient)
      LeftBound197040.bound (LeftBound197040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events769.exact197047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197040.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound197057.bound, LeftBound197040.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197057.bound, LeftBound197040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound197057.actual selector witness, LeftBound197040.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197228

namespace LeftBound197231
def owner : Owner := ⟨.program ⟨257⟩, ⟨70338⟩⟩
def transferEvent : Nat := 197231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 197225 .summary, .result 197047 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 197225 .summary)
      LeftBound197059.bound (LeftBound197059.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68120⟩⟩) (rawTerms := some (Proof.Events770.exact197225RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound197059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 197047 .summary)
      LeftBound197042.bound (LeftBound197042.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70337⟩⟩) (rawTerms := some (Proof.Events769.exact197047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound197042.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound197059.bound, LeftBound197042.bound]
def bound : CoeffClass := .finite ⟨32191361068277642793642192273408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197059.bound, LeftBound197042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound197059.actual selector witness, LeftBound197042.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197231

namespace LeftBound197255
def owner : Owner := ⟨.program ⟨257⟩, ⟨25515⟩⟩
def transferEvent : Nat := 197255
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 197253 .coefficient) (.predecessor 1 197254 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197253 .coefficient)
      LeftAuthority9276.bound (LeftAuthority9276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197254 .coefficient)
      LeftBound192901.bound (LeftBound192901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192901.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority9276.bound LeftBound192901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9276.bound, LeftBound192901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority9276.actual selector witness) * (LeftBound192901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound197255

namespace LeftBound197260
def owner : Owner := ⟨.program ⟨257⟩, ⟨8809⟩⟩
def transferEvent : Nat := 197260
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 197258 .coefficient) (.predecessor 1 197259 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197258 .coefficient)
      LeftBound192772.bound (LeftBound192772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192773RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197259 .coefficient)
      LeftBound21588.bound (LeftBound21588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound192772.bound LeftBound21588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192772.bound, LeftBound21588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound192772.actual selector witness) * (LeftBound21588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound197260

namespace LeftBound197265
def owner : Owner := ⟨.program ⟨257⟩, ⟨25516⟩⟩
def transferEvent : Nat := 197265
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 197263 .coefficient, .predecessor 1 197264 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197263 .coefficient)
      LeftBound197260.bound (LeftBound197260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197264 .coefficient)
      LeftBound197255.bound (LeftBound197255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound197260.bound, LeftBound197255.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197260.bound, LeftBound197255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound197260.actual selector witness, LeftBound197255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197265

namespace LeftBound197269
def owner : Owner := ⟨.program ⟨257⟩, ⟨25517⟩⟩
def transferEvent : Nat := 197269
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 197267 .coefficient, .predecessor 1 197268 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197267 .coefficient)
      LeftBound197265.bound (LeftBound197265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197268 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound197265.bound, LeftBound21580.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197265.bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound197265.actual selector witness, LeftBound21580.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound197269

namespace LeftBound197270
def owner : Owner := ⟨.program ⟨257⟩, ⟨25517⟩⟩
def transferEvent : Nat := 197270
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨101⟩⟩]⟩ [⟨.result 21581 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21581 .coefficient)
      LeftBound21580.bound (LeftBound21580.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨101⟩⟩) (rawTerms := some (Proof.Events084.exact21581RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21580.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21580.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21580.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound197270

namespace LeftBound197275
def owner : Owner := ⟨.program ⟨257⟩, ⟨62522⟩⟩
def transferEvent : Nat := 197275
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 197273 .coefficient) (.predecessor 1 197274 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 197273 .coefficient)
      LeftBound197269.bound (LeftBound197269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events770.exact197272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound197269.bound, RecordedBoundRefines] <;> decide)
      (LeftBound197269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 197274 .coefficient)
      LeftAuthority9279.bound (LeftAuthority9279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9279.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound197269.bound LeftAuthority9279.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound197269.bound, LeftAuthority9279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound197269.actual selector witness) * (LeftAuthority9279.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound197275

namespace LeftBound197276
def owner : Owner := ⟨.program ⟨257⟩, ⟨62522⟩⟩
def transferEvent : Nat := 197276
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨62519⟩⟩], []⟩ [⟨.result 9280 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 9280 .coefficient)
      LeftAuthority9279.bound (LeftAuthority9279.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨62519⟩⟩) (rawTerms := some (Proof.Events036.exact9280RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9279.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9279.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority9279.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound197276

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
