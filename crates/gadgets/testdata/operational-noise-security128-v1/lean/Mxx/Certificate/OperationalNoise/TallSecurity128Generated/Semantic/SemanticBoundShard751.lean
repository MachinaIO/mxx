import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard748
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard749
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard750

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound115228
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 115228
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115226 .coefficient, .predecessor 1 115227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115226 .coefficient)
      LeftBound115224.bound (LeftBound115224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115224.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115224.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115227 .coefficient)
      LeftAuthority115158.bound (LeftAuthority115158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115158.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115224.bound, LeftAuthority115158.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115224.bound, LeftAuthority115158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115224.actual selector witness, LeftAuthority115158.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115228

namespace LeftBound115232
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 115232
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115230 .coefficient, .predecessor 1 115231 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115230 .coefficient)
      LeftBound115228.bound (LeftBound115228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115228.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115231 .coefficient)
      LeftAuthority115155.bound (LeftAuthority115155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115228.bound, LeftAuthority115155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115228.bound, LeftAuthority115155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115228.actual selector witness, LeftAuthority115155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115232

namespace LeftBound115236
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 115236
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115234 .coefficient, .predecessor 1 115235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115234 .coefficient)
      LeftBound115232.bound (LeftBound115232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115235 .coefficient)
      LeftAuthority115152.bound (LeftAuthority115152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115152.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115232.bound, LeftAuthority115152.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115232.bound, LeftAuthority115152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115232.actual selector witness, LeftAuthority115152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115236

namespace LeftBound115240
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 115240
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115238 .coefficient, .predecessor 1 115239 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115238 .coefficient)
      LeftBound115236.bound (LeftBound115236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115239 .coefficient)
      LeftAuthority115149.bound (LeftAuthority115149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115149.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115149.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115236.bound, LeftAuthority115149.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115236.bound, LeftAuthority115149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115236.actual selector witness, LeftAuthority115149.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115240

namespace LeftBound115244
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 115244
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115242 .coefficient, .predecessor 1 115243 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115242 .coefficient)
      LeftBound115240.bound (LeftBound115240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115240.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115243 .coefficient)
      LeftAuthority115146.bound (LeftAuthority115146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115146.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115240.bound, LeftAuthority115146.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115240.bound, LeftAuthority115146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115240.actual selector witness, LeftAuthority115146.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115244

namespace LeftBound115248
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 115248
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115246 .coefficient, .predecessor 1 115247 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115246 .coefficient)
      LeftBound115244.bound (LeftBound115244.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115244.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115244.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115247 .coefficient)
      LeftAuthority115143.bound (LeftAuthority115143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115143.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115244.bound, LeftAuthority115143.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115244.bound, LeftAuthority115143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115244.actual selector witness, LeftAuthority115143.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115248

namespace LeftBound115252
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 115252
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115250 .coefficient, .predecessor 1 115251 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115250 .coefficient)
      LeftBound115248.bound (LeftBound115248.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115248.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115248.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115251 .coefficient)
      LeftAuthority115140.bound (LeftAuthority115140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115140.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115248.bound, LeftAuthority115140.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115248.bound, LeftAuthority115140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115248.actual selector witness, LeftAuthority115140.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115252

namespace LeftBound115256
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 115256
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115254 .coefficient, .predecessor 1 115255 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115254 .coefficient)
      LeftBound115252.bound (LeftBound115252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115255 .coefficient)
      LeftAuthority115137.bound (LeftAuthority115137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115137.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115137.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115252.bound, LeftAuthority115137.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115252.bound, LeftAuthority115137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115252.actual selector witness, LeftAuthority115137.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115256

namespace LeftBound115260
def owner : Owner := ⟨.program ⟨257⟩, ⟨69094⟩⟩
def transferEvent : Nat := 115260
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115258 .coefficient, .predecessor 1 115259 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115258 .coefficient)
      LeftBound115256.bound (LeftBound115256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115259 .coefficient)
      LeftBound115116.bound (LeftBound115116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115256.bound, LeftBound115116.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115256.bound, LeftBound115116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115256.actual selector witness, LeftBound115116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115260

namespace LeftBound115264
def owner : Owner := ⟨.program ⟨257⟩, ⟨71268⟩⟩
def transferEvent : Nat := 115264
def frameStart : Nat := 114586
def rule : BoundRule := .product (.predecessor 0 115262 .coefficient) (.predecessor 1 115263 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115262 .coefficient)
      LeftBound115260.bound (LeftBound115260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115263 .coefficient)
      LeftAuthority115101.bound (LeftAuthority115101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound115260.bound LeftAuthority115101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115260.bound, LeftAuthority115101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound115260.actual selector witness) * (LeftAuthority115101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115264

namespace LeftBound115343
def owner : Owner := ⟨.program ⟨257⟩, ⟨67478⟩⟩
def transferEvent : Nat := 115343
def frameStart : Nat := 114586
def rule : BoundRule := .product (.predecessor 0 115341 .coefficient) (.predecessor 1 115342 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115341 .coefficient)
      LeftAuthority115112.bound (LeftAuthority115112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events449.exact115113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115342 .coefficient)
      LeftAuthority115339.bound (LeftAuthority115339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115339.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115339.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority115112.bound LeftAuthority115339.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115112.bound, LeftAuthority115339.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority115112.actual selector witness) * (LeftAuthority115339.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115343

namespace LeftBound115351
def owner : Owner := ⟨.program ⟨257⟩, ⟨67482⟩⟩
def transferEvent : Nat := 115351
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115349 .coefficient, .predecessor 1 115350 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115349 .coefficient)
      LeftAuthority115347.bound (LeftAuthority115347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority115347.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority115347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115350 .coefficient)
      LeftBound115343.bound (LeftBound115343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115343.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority115347.bound, LeftBound115343.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority115347.bound, LeftBound115343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority115347.actual selector witness, LeftBound115343.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115351

namespace LeftBound115355
def owner : Owner := ⟨.program ⟨257⟩, ⟨71272⟩⟩
def transferEvent : Nat := 115355
def frameStart : Nat := 114586
def rule : BoundRule := .sum [.predecessor 0 115353 .coefficient, .predecessor 1 115354 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115353 .coefficient)
      LeftBound115351.bound (LeftBound115351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115354 .coefficient)
      LeftBound115264.bound (LeftBound115264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115264.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound115351.bound, LeftBound115264.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115351.bound, LeftBound115264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound115351.actual selector witness, LeftBound115264.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115355

namespace LeftBound115402
def owner : Owner := ⟨.program ⟨257⟩, ⟨71270⟩⟩
def transferEvent : Nat := 115402
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 115400 .coefficient, .predecessor 1 115401 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115400 .coefficient)
      LeftBound113993.bound (LeftBound113993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115399RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115401 .coefficient)
      LeftBound113908.bound (LeftBound113908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events445.exact113983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113993.bound, LeftBound113908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113993.bound, LeftBound113908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113993.actual selector witness, LeftBound113908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115402

namespace LeftBound115439
def owner : Owner := ⟨.program ⟨257⟩, ⟨71270⟩⟩
def transferEvent : Nat := 115439
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 115399 .summary, .result 113983 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 115399 .summary)
      LeftBound113995.bound (LeftBound113995.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨68383⟩⟩) (rawTerms := some (Proof.Events450.exact115399RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113983 .summary)
      LeftBound113910.bound (LeftBound113910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71269⟩⟩) (rawTerms := some (Proof.Events445.exact113983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113995.bound, LeftBound113910.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469506489977540968448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113995.bound, LeftBound113910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113995.actual selector witness, LeftBound113910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound115439

namespace LeftBound115443
def owner : Owner := ⟨.program ⟨257⟩, ⟨71271⟩⟩
def transferEvent : Nat := 115443
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 115441 .coefficient) (.predecessor 1 115442 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 115441 .coefficient)
      LeftBound115402.bound (LeftBound115402.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events450.exact115440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound115402.bound, RecordedBoundRefines] <;> decide)
      (LeftBound115402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 115442 .coefficient)
      LeftBound15521.bound (LeftBound15521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15521.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound115402.bound LeftBound15521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound115402.bound, LeftBound15521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound115402.actual selector witness) * (LeftBound15521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound115443

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
