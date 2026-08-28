import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1763

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound261305
def owner : Owner := ⟨.program ⟨257⟩, ⟨66252⟩⟩
def transferEvent : Nat := 261305
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261303 .coefficient, .predecessor 1 261304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261303 .coefficient)
      LeftBound261301.bound (LeftBound261301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261304 .coefficient)
      LeftAuthority261062.bound (LeftAuthority261062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact261063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261062.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261301.bound, LeftAuthority261062.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261301.bound, LeftAuthority261062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261301.actual selector witness, LeftAuthority261062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261305

namespace LeftBound261309
def owner : Owner := ⟨.program ⟨257⟩, ⟨66253⟩⟩
def transferEvent : Nat := 261309
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261307 .coefficient, .predecessor 1 261308 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261307 .coefficient)
      LeftBound261305.bound (LeftBound261305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261308 .coefficient)
      LeftAuthority261039.bound (LeftAuthority261039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact261040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261305.bound, LeftAuthority261039.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261305.bound, LeftAuthority261039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261305.actual selector witness, LeftAuthority261039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261309

namespace LeftBound261313
def owner : Owner := ⟨.program ⟨257⟩, ⟨66254⟩⟩
def transferEvent : Nat := 261313
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261311 .coefficient, .predecessor 1 261312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261311 .coefficient)
      LeftBound261309.bound (LeftBound261309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261312 .coefficient)
      LeftAuthority261016.bound (LeftAuthority261016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact261017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261309.bound, LeftAuthority261016.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261309.bound, LeftAuthority261016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261309.actual selector witness, LeftAuthority261016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261313

namespace LeftBound261317
def owner : Owner := ⟨.program ⟨257⟩, ⟨66255⟩⟩
def transferEvent : Nat := 261317
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261315 .coefficient, .predecessor 1 261316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261315 .coefficient)
      LeftBound261313.bound (LeftBound261313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261316 .coefficient)
      LeftAuthority260993.bound (LeftAuthority260993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260993.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261313.bound, LeftAuthority260993.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261313.bound, LeftAuthority260993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261313.actual selector witness, LeftAuthority260993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261317

namespace LeftBound261321
def owner : Owner := ⟨.program ⟨257⟩, ⟨66256⟩⟩
def transferEvent : Nat := 261321
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261319 .coefficient, .predecessor 1 261320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261319 .coefficient)
      LeftBound261317.bound (LeftBound261317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261320 .coefficient)
      LeftAuthority260970.bound (LeftAuthority260970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261317.bound, LeftAuthority260970.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261317.bound, LeftAuthority260970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261317.actual selector witness, LeftAuthority260970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261321

namespace LeftBound261325
def owner : Owner := ⟨.program ⟨257⟩, ⟨66257⟩⟩
def transferEvent : Nat := 261325
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261323 .coefficient, .predecessor 1 261324 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261323 .coefficient)
      LeftBound261321.bound (LeftBound261321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261324 .coefficient)
      LeftAuthority260947.bound (LeftAuthority260947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261321.bound, LeftAuthority260947.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261321.bound, LeftAuthority260947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261321.actual selector witness, LeftAuthority260947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261325

namespace LeftBound261329
def owner : Owner := ⟨.program ⟨257⟩, ⟨66258⟩⟩
def transferEvent : Nat := 261329
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261327 .coefficient, .predecessor 1 261328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261327 .coefficient)
      LeftBound261325.bound (LeftBound261325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261328 .coefficient)
      LeftAuthority260924.bound (LeftAuthority260924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261325.bound, LeftAuthority260924.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261325.bound, LeftAuthority260924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261325.actual selector witness, LeftAuthority260924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261329

namespace LeftBound261333
def owner : Owner := ⟨.program ⟨257⟩, ⟨66259⟩⟩
def transferEvent : Nat := 261333
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261331 .coefficient, .predecessor 1 261332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261331 .coefficient)
      LeftBound261329.bound (LeftBound261329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261332 .coefficient)
      LeftAuthority260901.bound (LeftAuthority260901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260901.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261329.bound, LeftAuthority260901.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261329.bound, LeftAuthority260901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261329.actual selector witness, LeftAuthority260901.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261333

namespace LeftBound261337
def owner : Owner := ⟨.program ⟨257⟩, ⟨66260⟩⟩
def transferEvent : Nat := 261337
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261335 .coefficient, .predecessor 1 261336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261335 .coefficient)
      LeftBound261333.bound (LeftBound261333.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261333.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261336 .coefficient)
      LeftAuthority260878.bound (LeftAuthority260878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1019.exact260879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority260878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority260878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261333.bound, LeftAuthority260878.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261333.bound, LeftAuthority260878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261333.actual selector witness, LeftAuthority260878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261337

namespace LeftBound261340
def owner : Owner := ⟨.program ⟨257⟩, ⟨66261⟩⟩
def transferEvent : Nat := 261340
def frameStart : Nat := 260836
def rule : BoundRule := .identity (.predecessor 0 261339 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261339 .coefficient)
      LeftBound261337.bound (LeftBound261337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261337.derived selector witness)

def rawBound : CoeffClass := LeftBound261337.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261337.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound261337.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound261340

namespace LeftBound261357
def owner : Owner := ⟨.program ⟨257⟩, ⟨69067⟩⟩
def transferEvent : Nat := 261357
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261355 .coefficient, .predecessor 1 261356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261355 .coefficient)
      LeftBound261340.bound (LeftBound261340.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound261340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261356 .coefficient)
      LeftAuthority261353.bound (LeftAuthority261353.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority261353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261340.bound, LeftAuthority261353.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261340.bound, LeftAuthority261353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261340.actual selector witness, LeftAuthority261353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261357

namespace LeftBound261360
def owner : Owner := ⟨.program ⟨257⟩, ⟨69068⟩⟩
def transferEvent : Nat := 261360
def frameStart : Nat := 260836
def rule : BoundRule := .identity (.predecessor 0 261359 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261359 .coefficient)
      LeftBound261357.bound (LeftBound261357.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound261357.derived selector witness)

def rawBound : CoeffClass := LeftBound261357.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound261357.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound261360

namespace LeftBound261366
def owner : Owner := ⟨.program ⟨257⟩, ⟨69069⟩⟩
def transferEvent : Nat := 261366
def frameStart : Nat := 260836
def rule : BoundRule := .product (.predecessor 0 261364 .coefficient) (.predecessor 1 261365 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261364 .coefficient)
      LeftAuthority261362.bound (LeftAuthority261362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261365 .coefficient)
      LeftBound261360.bound (LeftBound261360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1020.exact261361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority261362.bound LeftBound261360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261362.bound, LeftBound261360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority261362.actual selector witness) * (LeftBound261360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261366

namespace LeftBound261442
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 261442
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261440 .coefficient, .predecessor 1 261441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261440 .coefficient)
      LeftAuthority261438.bound (LeftAuthority261438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261438.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261441 .coefficient)
      LeftAuthority261435.bound (LeftAuthority261435.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261435.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261435.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority261438.bound, LeftAuthority261435.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261438.bound, LeftAuthority261435.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority261438.actual selector witness, LeftAuthority261435.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261442

namespace LeftBound261446
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 261446
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261444 .coefficient, .predecessor 1 261445 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261444 .coefficient)
      LeftBound261442.bound (LeftBound261442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261445 .coefficient)
      LeftAuthority261432.bound (LeftAuthority261432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261442.bound, LeftAuthority261432.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261442.bound, LeftAuthority261432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261442.actual selector witness, LeftAuthority261432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261446

namespace LeftBound261450
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 261450
def frameStart : Nat := 260836
def rule : BoundRule := .sum [.predecessor 0 261448 .coefficient, .predecessor 1 261449 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261448 .coefficient)
      LeftBound261446.bound (LeftBound261446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261446.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261446.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261449 .coefficient)
      LeftAuthority261429.bound (LeftAuthority261429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1021.exact261430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261429.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261446.bound, LeftAuthority261429.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261446.bound, LeftAuthority261429.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261446.actual selector witness, LeftAuthority261429.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261450

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
