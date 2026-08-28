import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1948

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound287335
def owner : Owner := ⟨.program ⟨257⟩, ⟨51049⟩⟩
def transferEvent : Nat := 287335
def frameStart : Nat := 287247
def rule : BoundRule := .product (.predecessor 0 287333 .coefficient) (.predecessor 1 287334 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287333 .coefficient)
      LeftAuthority287308.bound (LeftAuthority287308.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287309RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287308.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287334 .coefficient)
      LeftAuthority287331.bound (LeftAuthority287331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287331.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287331.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority287308.bound LeftAuthority287331.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287308.bound, LeftAuthority287331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority287308.actual selector witness) * (LeftAuthority287331.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287335

namespace LeftBound287343
def owner : Owner := ⟨.program ⟨257⟩, ⟨51050⟩⟩
def transferEvent : Nat := 287343
def frameStart : Nat := 287247
def rule : BoundRule := .sum [.predecessor 0 287341 .coefficient, .predecessor 1 287342 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287341 .coefficient)
      LeftAuthority287339.bound (LeftAuthority287339.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority287339.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority287339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287342 .coefficient)
      LeftBound287335.bound (LeftBound287335.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287337RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287335.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287335.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority287339.bound, LeftBound287335.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority287339.bound, LeftBound287335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority287339.actual selector witness, LeftBound287335.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287343

namespace LeftBound287347
def owner : Owner := ⟨.program ⟨257⟩, ⟨52771⟩⟩
def transferEvent : Nat := 287347
def frameStart : Nat := 287247
def rule : BoundRule := .sum [.predecessor 0 287345 .coefficient, .predecessor 1 287346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287345 .coefficient)
      LeftBound287343.bound (LeftBound287343.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287343.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287346 .coefficient)
      LeftBound287324.bound (LeftBound287324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287324.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287324.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287343.bound, LeftBound287324.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287343.bound, LeftBound287324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287343.actual selector witness, LeftBound287324.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287347

namespace LeftBound287360
def owner : Owner := ⟨.program ⟨257⟩, ⟨52769⟩⟩
def transferEvent : Nat := 287360
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 287358 .coefficient, .predecessor 1 287359 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287358 .coefficient)
      LeftBound287189.bound (LeftBound287189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287359 .coefficient)
      LeftBound287172.bound (LeftBound287172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1121.exact287179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287172.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287189.bound, LeftBound287172.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287189.bound, LeftBound287172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287189.actual selector witness, LeftBound287172.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287360

namespace LeftBound287363
def owner : Owner := ⟨.program ⟨257⟩, ⟨52769⟩⟩
def transferEvent : Nat := 287363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 287357 .summary, .result 287179 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287357 .summary)
      LeftBound287191.bound (LeftBound287191.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51639⟩⟩) (rawTerms := some (Proof.Events1122.exact287357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287179 .summary)
      LeftBound287174.bound (LeftBound287174.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52768⟩⟩) (rawTerms := some (Proof.Events1121.exact287179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287191.bound, LeftBound287174.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287191.bound, LeftBound287174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287191.actual selector witness, LeftBound287174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287363

namespace LeftBound287387
def owner : Owner := ⟨.program ⟨257⟩, ⟨24219⟩⟩
def transferEvent : Nat := 287387
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 287385 .coefficient) (.predecessor 1 287386 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287385 .coefficient)
      LeftAuthority13873.bound (LeftAuthority13873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13873.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287386 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13873.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13873.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13873.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound287387

namespace LeftBound287392
def owner : Owner := ⟨.program ⟨257⟩, ⟨7929⟩⟩
def transferEvent : Nat := 287392
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 287390 .coefficient) (.predecessor 1 287391 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287390 .coefficient)
      LeftBound280522.bound (LeftBound280522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287391 .coefficient)
      LeftBound24093.bound (LeftBound24093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound280522.bound LeftBound24093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280522.bound, LeftBound24093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound280522.actual selector witness) * (LeftBound24093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287392

namespace LeftBound287397
def owner : Owner := ⟨.program ⟨257⟩, ⟨24220⟩⟩
def transferEvent : Nat := 287397
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 287395 .coefficient, .predecessor 1 287396 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287395 .coefficient)
      LeftBound287392.bound (LeftBound287392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287394RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287396 .coefficient)
      LeftBound287387.bound (LeftBound287387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287392.bound, LeftBound287387.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287392.bound, LeftBound287387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287392.actual selector witness, LeftBound287387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287397

namespace LeftBound287401
def owner : Owner := ⟨.program ⟨257⟩, ⟨24221⟩⟩
def transferEvent : Nat := 287401
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 287399 .coefficient, .predecessor 1 287400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287399 .coefficient)
      LeftBound287397.bound (LeftBound287397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287400 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287397.bound, LeftBound24085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287397.bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287397.actual selector witness, LeftBound24085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287401

namespace LeftBound287402
def owner : Owner := ⟨.program ⟨257⟩, ⟨24221⟩⟩
def transferEvent : Nat := 287402
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩ [⟨.result 24086 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24086 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨133⟩⟩) (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24085.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24085.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound287402

namespace LeftBound287407
def owner : Owner := ⟨.program ⟨257⟩, ⟨31326⟩⟩
def transferEvent : Nat := 287407
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 287405 .coefficient) (.predecessor 1 287406 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287405 .coefficient)
      LeftBound287401.bound (LeftBound287401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287401.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287406 .coefficient)
      LeftAuthority13876.bound (LeftAuthority13876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13876.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound287401.bound LeftAuthority13876.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287401.bound, LeftAuthority13876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound287401.actual selector witness) * (LeftAuthority13876.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287407

namespace LeftBound287408
def owner : Owner := ⟨.program ⟨257⟩, ⟨31326⟩⟩
def transferEvent : Nat := 287408
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨31323⟩⟩], []⟩ [⟨.result 13877 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 13877 .coefficient)
      LeftAuthority13876.bound (LeftAuthority13876.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨31323⟩⟩) (rawTerms := some (Proof.Events054.exact13877RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13876.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13876.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority13876.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound287408

namespace LeftBound287409
def owner : Owner := ⟨.program ⟨257⟩, ⟨31326⟩⟩
def transferEvent : Nat := 287409
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 287404 .summary) (.transfer 287408) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 287404 .summary)
      LeftBound287402.bound (LeftBound287402.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24221⟩⟩) (rawTerms := some (Proof.Events1122.exact287404RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound287402.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 287408)
      LeftBound287408.bound (LeftBound287408.actual selector witness) := by
  exact .transfer (LeftBound287408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound287402.bound LeftBound287408.bound
def bound : CoeffClass := .finite ⟨5111808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287402.bound, LeftBound287408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound287402.actual selector witness) * (LeftBound287408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287409

namespace LeftBound287415
def owner : Owner := ⟨.program ⟨257⟩, ⟨31327⟩⟩
def transferEvent : Nat := 287415
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 287413 .coefficient) (.predecessor 1 287414 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287413 .coefficient)
      LeftAuthority13876.bound (LeftAuthority13876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287414 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13876.bound LeftBound280651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13876.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13876.actual selector witness) * (LeftBound280651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound287415

namespace LeftBound287420
def owner : Owner := ⟨.program ⟨257⟩, ⟨7909⟩⟩
def transferEvent : Nat := 287420
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 287418 .coefficient) (.predecessor 1 287419 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287418 .coefficient)
      LeftBound280522.bound (LeftBound280522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287419 .coefficient)
      LeftBound24134.bound (LeftBound24134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24134.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound280522.bound LeftBound24134.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound280522.bound, LeftBound24134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound280522.actual selector witness) * (LeftBound24134.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound287420

namespace LeftBound287425
def owner : Owner := ⟨.program ⟨257⟩, ⟨31328⟩⟩
def transferEvent : Nat := 287425
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 287423 .coefficient, .predecessor 1 287424 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 287423 .coefficient)
      LeftBound287420.bound (LeftBound287420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287420.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287420.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 287424 .coefficient)
      LeftBound287415.bound (LeftBound287415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1122.exact287417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound287415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound287415.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound287420.bound, LeftBound287415.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound287420.bound, LeftBound287415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound287420.actual selector witness, LeftBound287415.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound287425

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
