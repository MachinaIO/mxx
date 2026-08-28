import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound144273
def owner : Owner := ⟨.program ⟨257⟩, ⟨18734⟩⟩
def transferEvent : Nat := 144273
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144271 .coefficient, .predecessor 1 144272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144271 .coefficient)
      LeftAuthority144269.bound (LeftAuthority144269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144272 .coefficient)
      LeftAuthority144246.bound (LeftAuthority144246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority144269.bound, LeftAuthority144246.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority144269.bound, LeftAuthority144246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority144269.actual selector witness, LeftAuthority144246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144273

namespace LeftBound144277
def owner : Owner := ⟨.program ⟨257⟩, ⟨21954⟩⟩
def transferEvent : Nat := 144277
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144275 .coefficient, .predecessor 1 144276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144275 .coefficient)
      LeftBound144273.bound (LeftBound144273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144276 .coefficient)
      LeftAuthority144223.bound (LeftAuthority144223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144223.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144273.bound, LeftAuthority144223.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144273.bound, LeftAuthority144223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144273.actual selector witness, LeftAuthority144223.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144277

namespace LeftBound144281
def owner : Owner := ⟨.program ⟨257⟩, ⟨31974⟩⟩
def transferEvent : Nat := 144281
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144279 .coefficient, .predecessor 1 144280 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144279 .coefficient)
      LeftBound144277.bound (LeftBound144277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144280 .coefficient)
      LeftAuthority144200.bound (LeftAuthority144200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144200.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144277.bound, LeftAuthority144200.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144277.bound, LeftAuthority144200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144277.actual selector witness, LeftAuthority144200.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144281

namespace LeftBound144285
def owner : Owner := ⟨.program ⟨257⟩, ⟨51029⟩⟩
def transferEvent : Nat := 144285
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144283 .coefficient, .predecessor 1 144284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144283 .coefficient)
      LeftBound144281.bound (LeftBound144281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144281.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144284 .coefficient)
      LeftAuthority144177.bound (LeftAuthority144177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144281.bound, LeftAuthority144177.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144281.bound, LeftAuthority144177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144281.actual selector witness, LeftAuthority144177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144285

namespace LeftBound144289
def owner : Owner := ⟨.program ⟨257⟩, ⟨54009⟩⟩
def transferEvent : Nat := 144289
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144287 .coefficient, .predecessor 1 144288 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144287 .coefficient)
      LeftBound144285.bound (LeftBound144285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144288 .coefficient)
      LeftAuthority144154.bound (LeftAuthority144154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144154.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144285.bound, LeftAuthority144154.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144285.bound, LeftAuthority144154.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144285.actual selector witness, LeftAuthority144154.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144289

namespace LeftBound144293
def owner : Owner := ⟨.program ⟨257⟩, ⟨56989⟩⟩
def transferEvent : Nat := 144293
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144291 .coefficient, .predecessor 1 144292 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144291 .coefficient)
      LeftBound144289.bound (LeftBound144289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144292 .coefficient)
      LeftAuthority144131.bound (LeftAuthority144131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144131.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144131.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144289.bound, LeftAuthority144131.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144289.bound, LeftAuthority144131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144289.actual selector witness, LeftAuthority144131.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144293

namespace LeftBound144297
def owner : Owner := ⟨.program ⟨257⟩, ⟨59969⟩⟩
def transferEvent : Nat := 144297
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144295 .coefficient, .predecessor 1 144296 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144295 .coefficient)
      LeftBound144293.bound (LeftBound144293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144296 .coefficient)
      LeftAuthority144108.bound (LeftAuthority144108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact144109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144108.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144108.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144293.bound, LeftAuthority144108.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144293.bound, LeftAuthority144108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144293.actual selector witness, LeftAuthority144108.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144297

namespace LeftBound144301
def owner : Owner := ⟨.program ⟨257⟩, ⟨62949⟩⟩
def transferEvent : Nat := 144301
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144299 .coefficient, .predecessor 1 144300 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144299 .coefficient)
      LeftBound144297.bound (LeftBound144297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144300 .coefficient)
      LeftAuthority144085.bound (LeftAuthority144085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact144086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144297.bound, LeftAuthority144085.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144297.bound, LeftAuthority144085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144297.actual selector witness, LeftAuthority144085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144301

namespace LeftBound144305
def owner : Owner := ⟨.program ⟨257⟩, ⟨66112⟩⟩
def transferEvent : Nat := 144305
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144303 .coefficient, .predecessor 1 144304 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144303 .coefficient)
      LeftBound144301.bound (LeftBound144301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144302RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144304 .coefficient)
      LeftAuthority144062.bound (LeftAuthority144062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact144063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144062.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144301.bound, LeftAuthority144062.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144301.bound, LeftAuthority144062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144301.actual selector witness, LeftAuthority144062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144305

namespace LeftBound144309
def owner : Owner := ⟨.program ⟨257⟩, ⟨66113⟩⟩
def transferEvent : Nat := 144309
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144307 .coefficient, .predecessor 1 144308 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144307 .coefficient)
      LeftBound144305.bound (LeftBound144305.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144306RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144305.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144305.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144308 .coefficient)
      LeftAuthority144039.bound (LeftAuthority144039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact144040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144039.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144039.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144305.bound, LeftAuthority144039.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144305.bound, LeftAuthority144039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144305.actual selector witness, LeftAuthority144039.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144309

namespace LeftBound144313
def owner : Owner := ⟨.program ⟨257⟩, ⟨66114⟩⟩
def transferEvent : Nat := 144313
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144311 .coefficient, .predecessor 1 144312 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144311 .coefficient)
      LeftBound144309.bound (LeftBound144309.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144310RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144309.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144309.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144312 .coefficient)
      LeftAuthority144016.bound (LeftAuthority144016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact144017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority144016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority144016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144309.bound, LeftAuthority144016.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144309.bound, LeftAuthority144016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144309.actual selector witness, LeftAuthority144016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144313

namespace LeftBound144317
def owner : Owner := ⟨.program ⟨257⟩, ⟨66115⟩⟩
def transferEvent : Nat := 144317
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144315 .coefficient, .predecessor 1 144316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144315 .coefficient)
      LeftBound144313.bound (LeftBound144313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144316 .coefficient)
      LeftAuthority143993.bound (LeftAuthority143993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143993.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143993.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144313.bound, LeftAuthority143993.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144313.bound, LeftAuthority143993.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144313.actual selector witness, LeftAuthority143993.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144317

namespace LeftBound144321
def owner : Owner := ⟨.program ⟨257⟩, ⟨66116⟩⟩
def transferEvent : Nat := 144321
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144319 .coefficient, .predecessor 1 144320 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144319 .coefficient)
      LeftBound144317.bound (LeftBound144317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144320 .coefficient)
      LeftAuthority143970.bound (LeftAuthority143970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143970.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144317.bound, LeftAuthority143970.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144317.bound, LeftAuthority143970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144317.actual selector witness, LeftAuthority143970.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144321

namespace LeftBound144325
def owner : Owner := ⟨.program ⟨257⟩, ⟨66117⟩⟩
def transferEvent : Nat := 144325
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144323 .coefficient, .predecessor 1 144324 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144323 .coefficient)
      LeftBound144321.bound (LeftBound144321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144324 .coefficient)
      LeftAuthority143947.bound (LeftAuthority143947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144321.bound, LeftAuthority143947.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144321.bound, LeftAuthority143947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144321.actual selector witness, LeftAuthority143947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144325

namespace LeftBound144329
def owner : Owner := ⟨.program ⟨257⟩, ⟨66118⟩⟩
def transferEvent : Nat := 144329
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144327 .coefficient, .predecessor 1 144328 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144327 .coefficient)
      LeftBound144325.bound (LeftBound144325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144326RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144325.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144328 .coefficient)
      LeftAuthority143924.bound (LeftAuthority143924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144325.bound, LeftAuthority143924.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144325.bound, LeftAuthority143924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144325.actual selector witness, LeftAuthority143924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144329

namespace LeftBound144333
def owner : Owner := ⟨.program ⟨257⟩, ⟨66119⟩⟩
def transferEvent : Nat := 144333
def frameStart : Nat := 143836
def rule : BoundRule := .sum [.predecessor 0 144331 .coefficient, .predecessor 1 144332 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 144331 .coefficient)
      LeftBound144329.bound (LeftBound144329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events563.exact144330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound144329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound144329.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 144332 .coefficient)
      LeftAuthority143901.bound (LeftAuthority143901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events562.exact143902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143901.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound144329.bound, LeftAuthority143901.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound144329.bound, LeftAuthority143901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound144329.actual selector witness, LeftAuthority143901.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound144333

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
