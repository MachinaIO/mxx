import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard034
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard035

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound11235
def owner : Owner := ⟨.program ⟨257⟩, ⟨63068⟩⟩
def transferEvent : Nat := 11235
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11233 .coefficient, .predecessor 1 11234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11233 .coefficient)
      LeftBound11231.bound (LeftBound11231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11234 .coefficient)
      LeftBound11134.bound (LeftBound11134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11231.bound, LeftBound11134.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11231.bound, LeftBound11134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11231.actual selector witness, LeftBound11134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11235

namespace LeftBound11239
def owner : Owner := ⟨.program ⟨257⟩, ⟨66520⟩⟩
def transferEvent : Nat := 11239
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11237 .coefficient, .predecessor 1 11238 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11237 .coefficient)
      LeftBound11235.bound (LeftBound11235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11238 .coefficient)
      LeftBound11126.bound (LeftBound11126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11235.bound, LeftBound11126.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11235.bound, LeftBound11126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11235.actual selector witness, LeftBound11126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11239

namespace LeftBound11243
def owner : Owner := ⟨.program ⟨257⟩, ⟨66521⟩⟩
def transferEvent : Nat := 11243
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11241 .coefficient, .predecessor 1 11242 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11241 .coefficient)
      LeftBound11239.bound (LeftBound11239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11239.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11242 .coefficient)
      LeftBound11118.bound (LeftBound11118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11118.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11239.bound, LeftBound11118.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11239.bound, LeftBound11118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11239.actual selector witness, LeftBound11118.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11243

namespace LeftBound11247
def owner : Owner := ⟨.program ⟨257⟩, ⟨66522⟩⟩
def transferEvent : Nat := 11247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11245 .coefficient, .predecessor 1 11246 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11245 .coefficient)
      LeftBound11243.bound (LeftBound11243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11243.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11246 .coefficient)
      LeftBound11110.bound (LeftBound11110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11110.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11243.bound, LeftBound11110.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11243.bound, LeftBound11110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11243.actual selector witness, LeftBound11110.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11247

namespace LeftBound11251
def owner : Owner := ⟨.program ⟨257⟩, ⟨66523⟩⟩
def transferEvent : Nat := 11251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11249 .coefficient, .predecessor 1 11250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11249 .coefficient)
      LeftBound11247.bound (LeftBound11247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11250 .coefficient)
      LeftBound11102.bound (LeftBound11102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11247.bound, LeftBound11102.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11247.bound, LeftBound11102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11247.actual selector witness, LeftBound11102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11251

namespace LeftBound11255
def owner : Owner := ⟨.program ⟨257⟩, ⟨66524⟩⟩
def transferEvent : Nat := 11255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11253 .coefficient, .predecessor 1 11254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11253 .coefficient)
      LeftBound11251.bound (LeftBound11251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11254 .coefficient)
      LeftBound11094.bound (LeftBound11094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11251.bound, LeftBound11094.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11251.bound, LeftBound11094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11251.actual selector witness, LeftBound11094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11255

namespace LeftBound11259
def owner : Owner := ⟨.program ⟨257⟩, ⟨66525⟩⟩
def transferEvent : Nat := 11259
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11257 .coefficient, .predecessor 1 11258 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11257 .coefficient)
      LeftBound11255.bound (LeftBound11255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11258 .coefficient)
      LeftBound11086.bound (LeftBound11086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11086.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11086.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11255.bound, LeftBound11086.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11255.bound, LeftBound11086.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11255.actual selector witness, LeftBound11086.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11259

namespace LeftBound11263
def owner : Owner := ⟨.program ⟨257⟩, ⟨66526⟩⟩
def transferEvent : Nat := 11263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11261 .coefficient, .predecessor 1 11262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11261 .coefficient)
      LeftBound11259.bound (LeftBound11259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11259.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11259.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11262 .coefficient)
      LeftBound11078.bound (LeftBound11078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11259.bound, LeftBound11078.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11259.bound, LeftBound11078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11259.actual selector witness, LeftBound11078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11263

namespace LeftBound11267
def owner : Owner := ⟨.program ⟨257⟩, ⟨66527⟩⟩
def transferEvent : Nat := 11267
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11265 .coefficient, .predecessor 1 11266 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11265 .coefficient)
      LeftBound11263.bound (LeftBound11263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11266 .coefficient)
      LeftBound11070.bound (LeftBound11070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11070.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11263.bound, LeftBound11070.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11263.bound, LeftBound11070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11263.actual selector witness, LeftBound11070.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11267

namespace LeftBound11271
def owner : Owner := ⟨.program ⟨257⟩, ⟨66528⟩⟩
def transferEvent : Nat := 11271
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11269 .coefficient, .predecessor 1 11270 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11269 .coefficient)
      LeftBound11267.bound (LeftBound11267.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11267.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11267.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11270 .coefficient)
      LeftBound11062.bound (LeftBound11062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11267.bound, LeftBound11062.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11267.bound, LeftBound11062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11267.actual selector witness, LeftBound11062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11271

namespace LeftBound11275
def owner : Owner := ⟨.program ⟨257⟩, ⟨67440⟩⟩
def transferEvent : Nat := 11275
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 11273 .coefficient, .predecessor 1 11274 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11273 .coefficient)
      LeftBound11271.bound (LeftBound11271.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11271.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11271.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11274 .coefficient)
      LeftBound11054.bound (LeftBound11054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound11271.bound, LeftBound11054.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11271.bound, LeftBound11054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound11271.actual selector witness, LeftBound11054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound11275

namespace LeftBound11279
def owner : Owner := ⟨.program ⟨257⟩, ⟨67441⟩⟩
def transferEvent : Nat := 11279
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11277 .coefficient) (.predecessor 1 11278 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11277 .coefficient)
      LeftBound11275.bound (LeftBound11275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events044.exact11276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11275.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11275.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11278 .coefficient)
      LeftAuthority10552.bound (LeftAuthority10552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound11275.bound LeftAuthority10552.bound
def bound : CoeffClass := .finite ⟨245865348487757785284537663620193902876720873429212870896784286666958220519558789849412843015036898597232497788427963275816035035589061636259485163493584216607441867138949543880668860134181342838784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11275.bound, LeftAuthority10552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound11275.actual selector witness) * (LeftAuthority10552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11279

namespace LeftBound11802
def owner : Owner := ⟨.program ⟨257⟩, ⟨67418⟩⟩
def transferEvent : Nat := 11802
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11800 .coefficient) (.predecessor 1 11801 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11800 .coefficient)
      LeftAuthority11798.bound (LeftAuthority11798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11801 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11798.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11798.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority11798.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11802

namespace LeftBound11810
def owner : Owner := ⟨.program ⟨257⟩, ⟨48334⟩⟩
def transferEvent : Nat := 11810
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11808 .coefficient) (.predecessor 1 11809 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11808 .coefficient)
      LeftAuthority11806.bound (LeftAuthority11806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11809 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11806.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11806.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority11806.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11810

namespace LeftBound11818
def owner : Owner := ⟨.program ⟨257⟩, ⟨45654⟩⟩
def transferEvent : Nat := 11818
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11816 .coefficient) (.predecessor 1 11817 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11816 .coefficient)
      LeftAuthority11814.bound (LeftAuthority11814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11817 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11814.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11814.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority11814.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11818

namespace LeftBound11826
def owner : Owner := ⟨.program ⟨257⟩, ⟨42977⟩⟩
def transferEvent : Nat := 11826
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 11824 .coefficient) (.predecessor 1 11825 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 11824 .coefficient)
      LeftAuthority11822.bound (LeftAuthority11822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events046.exact11823RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11822.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 11825 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority11822.bound LeftAuthority562.bound
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11822.bound, LeftAuthority562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority11822.actual selector witness) * (LeftAuthority562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound11826

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
