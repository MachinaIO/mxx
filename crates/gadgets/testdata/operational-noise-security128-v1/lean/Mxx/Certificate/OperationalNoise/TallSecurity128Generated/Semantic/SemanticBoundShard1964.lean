import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1927
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1931
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1934
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1938
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1941
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1942
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1945
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1949
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1952
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1956
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1963

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound289293
def owner : Owner := ⟨.program ⟨257⟩, ⟨23690⟩⟩
def transferEvent : Nat := 289293
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289289 .summary, .result 288324 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289289 .summary)
      LeftBound289288.bound (LeftBound289288.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20470⟩⟩) (rawTerms := some (Proof.Events1130.exact289289RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289288.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 288324 .summary)
      LeftBound288323.bound (LeftBound288323.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23689⟩⟩) (rawTerms := some (Proof.Events1126.exact288324RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound288323.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289288.bound, LeftBound288323.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289288.bound, LeftBound288323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289288.actual selector witness, LeftBound288323.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289293

namespace LeftBound289297
def owner : Owner := ⟨.program ⟨257⟩, ⟨33710⟩⟩
def transferEvent : Nat := 289297
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289295 .coefficient, .predecessor 1 289296 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289295 .coefficient)
      LeftBound289292.bound (LeftBound289292.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289292.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289292.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289296 .coefficient)
      LeftBound287840.bound (LeftBound287840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1124.exact287844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289292.bound, LeftBound287840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289292.bound, LeftBound287840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289292.actual selector witness, LeftBound287840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289297

namespace LeftBound289298
def owner : Owner := ⟨.program ⟨257⟩, ⟨33710⟩⟩
def transferEvent : Nat := 289298
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289294 .summary, .result 287844 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289294 .summary)
      LeftBound289293.bound (LeftBound289293.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23690⟩⟩) (rawTerms := some (Proof.Events1130.exact289294RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287844 .summary)
      LeftBound287843.bound (LeftBound287843.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33709⟩⟩) (rawTerms := some (Proof.Events1124.exact287844RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289293.bound, LeftBound287843.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289293.bound, LeftBound287843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289293.actual selector witness, LeftBound287843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289298

namespace LeftBound289302
def owner : Owner := ⟨.program ⟨257⟩, ⟨52770⟩⟩
def transferEvent : Nat := 289302
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289300 .coefficient, .predecessor 1 289301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289300 .coefficient)
      LeftBound289297.bound (LeftBound289297.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289297.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289297.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289301 .coefficient)
      LeftBound287360.bound (LeftBound287360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287360.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289297.bound, LeftBound287360.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289297.bound, LeftBound287360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289297.actual selector witness, LeftBound287360.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289302

namespace LeftBound289303
def owner : Owner := ⟨.program ⟨257⟩, ⟨52770⟩⟩
def transferEvent : Nat := 289303
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289299 .summary, .result 287364 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289299 .summary)
      LeftBound289298.bound (LeftBound289298.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33710⟩⟩) (rawTerms := some (Proof.Events1130.exact289299RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287364 .summary)
      LeftBound287363.bound (LeftBound287363.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52769⟩⟩) (rawTerms := some (Proof.Events1122.exact287364RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289298.bound, LeftBound287363.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289298.bound, LeftBound287363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289298.actual selector witness, LeftBound287363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289303

namespace LeftBound289307
def owner : Owner := ⟨.program ⟨257⟩, ⟨55750⟩⟩
def transferEvent : Nat := 289307
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289305 .coefficient, .predecessor 1 289306 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289305 .coefficient)
      LeftBound289302.bound (LeftBound289302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289302.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289306 .coefficient)
      LeftBound286880.bound (LeftBound286880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1120.exact286884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286880.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289302.bound, LeftBound286880.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289302.bound, LeftBound286880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289302.actual selector witness, LeftBound286880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289307

namespace LeftBound289308
def owner : Owner := ⟨.program ⟨257⟩, ⟨55750⟩⟩
def transferEvent : Nat := 289308
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289304 .summary, .result 286884 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289304 .summary)
      LeftBound289303.bound (LeftBound289303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52770⟩⟩) (rawTerms := some (Proof.Events1130.exact289304RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286884 .summary)
      LeftBound286883.bound (LeftBound286883.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55749⟩⟩) (rawTerms := some (Proof.Events1120.exact286884RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289303.bound, LeftBound286883.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289303.bound, LeftBound286883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289303.actual selector witness, LeftBound286883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289308

namespace LeftBound289312
def owner : Owner := ⟨.program ⟨257⟩, ⟨58730⟩⟩
def transferEvent : Nat := 289312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289310 .coefficient, .predecessor 1 289311 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289310 .coefficient)
      LeftBound289307.bound (LeftBound289307.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289307.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289307.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289311 .coefficient)
      LeftBound286400.bound (LeftBound286400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1118.exact286404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound286400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound286400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289307.bound, LeftBound286400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289307.bound, LeftBound286400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289307.actual selector witness, LeftBound286400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289312

namespace LeftBound289313
def owner : Owner := ⟨.program ⟨257⟩, ⟨58730⟩⟩
def transferEvent : Nat := 289313
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289309 .summary, .result 286404 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289309 .summary)
      LeftBound289308.bound (LeftBound289308.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55750⟩⟩) (rawTerms := some (Proof.Events1130.exact289309RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 286404 .summary)
      LeftBound286403.bound (LeftBound286403.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58729⟩⟩) (rawTerms := some (Proof.Events1118.exact286404RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound286403.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289308.bound, LeftBound286403.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289308.bound, LeftBound286403.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289308.actual selector witness, LeftBound286403.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289313

namespace LeftBound289317
def owner : Owner := ⟨.program ⟨257⟩, ⟨61710⟩⟩
def transferEvent : Nat := 289317
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289315 .coefficient, .predecessor 1 289316 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289315 .coefficient)
      LeftBound289312.bound (LeftBound289312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289312.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289316 .coefficient)
      LeftBound285920.bound (LeftBound285920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1116.exact285924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289312.bound, LeftBound285920.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289312.bound, LeftBound285920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289312.actual selector witness, LeftBound285920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289317

namespace LeftBound289318
def owner : Owner := ⟨.program ⟨257⟩, ⟨61710⟩⟩
def transferEvent : Nat := 289318
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289314 .summary, .result 285924 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289314 .summary)
      LeftBound289313.bound (LeftBound289313.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58730⟩⟩) (rawTerms := some (Proof.Events1130.exact289314RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285924 .summary)
      LeftBound285923.bound (LeftBound285923.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61709⟩⟩) (rawTerms := some (Proof.Events1116.exact285924RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289313.bound, LeftBound285923.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289313.bound, LeftBound285923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289313.actual selector witness, LeftBound285923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289318

namespace LeftBound289322
def owner : Owner := ⟨.program ⟨257⟩, ⟨64690⟩⟩
def transferEvent : Nat := 289322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289320 .coefficient, .predecessor 1 289321 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289320 .coefficient)
      LeftBound289317.bound (LeftBound289317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289317.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289321 .coefficient)
      LeftBound285440.bound (LeftBound285440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1115.exact285444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound285440.bound, RecordedBoundRefines] <;> decide)
      (LeftBound285440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289317.bound, LeftBound285440.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289317.bound, LeftBound285440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289317.actual selector witness, LeftBound285440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289322

namespace LeftBound289323
def owner : Owner := ⟨.program ⟨257⟩, ⟨64690⟩⟩
def transferEvent : Nat := 289323
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289319 .summary, .result 285444 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289319 .summary)
      LeftBound289318.bound (LeftBound289318.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61710⟩⟩) (rawTerms := some (Proof.Events1130.exact289319RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 285444 .summary)
      LeftBound285443.bound (LeftBound285443.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64689⟩⟩) (rawTerms := some (Proof.Events1115.exact285444RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound285443.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289318.bound, LeftBound285443.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289318.bound, LeftBound285443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289318.actual selector witness, LeftBound285443.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289323

namespace LeftBound289327
def owner : Owner := ⟨.program ⟨257⟩, ⟨69707⟩⟩
def transferEvent : Nat := 289327
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289325 .coefficient, .predecessor 1 289326 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289325 .coefficient)
      LeftBound289322.bound (LeftBound289322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289326 .coefficient)
      LeftBound284960.bound (LeftBound284960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1113.exact284964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound284960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound284960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289322.bound, LeftBound284960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289322.bound, LeftBound284960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289322.actual selector witness, LeftBound284960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289327

namespace LeftBound289328
def owner : Owner := ⟨.program ⟨257⟩, ⟨69707⟩⟩
def transferEvent : Nat := 289328
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289324 .summary, .result 284964 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289324 .summary)
      LeftBound289323.bound (LeftBound289323.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64690⟩⟩) (rawTerms := some (Proof.Events1130.exact289324RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 284964 .summary)
      LeftBound284963.bound (LeftBound284963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69706⟩⟩) (rawTerms := some (Proof.Events1113.exact284964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound284963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289323.bound, LeftBound284963.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289323.bound, LeftBound284963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289323.actual selector witness, LeftBound284963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289328

namespace LeftBound289332
def owner : Owner := ⟨.program ⟨257⟩, ⟨69708⟩⟩
def transferEvent : Nat := 289332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289330 .coefficient, .predecessor 1 289331 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289330 .coefficient)
      LeftBound289327.bound (LeftBound289327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289327.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289331 .coefficient)
      LeftBound284480.bound (LeftBound284480.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1111.exact284484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound284480.bound, RecordedBoundRefines] <;> decide)
      (LeftBound284480.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289327.bound, LeftBound284480.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289327.bound, LeftBound284480.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289327.actual selector witness, LeftBound284480.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289332

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
