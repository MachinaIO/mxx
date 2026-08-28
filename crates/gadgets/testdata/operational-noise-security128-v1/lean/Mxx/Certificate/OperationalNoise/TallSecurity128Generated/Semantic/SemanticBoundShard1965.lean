import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1902
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1905
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1909
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1912
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1913
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1916
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1920
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1923
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1927
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1964

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound289333
def owner : Owner := ⟨.program ⟨257⟩, ⟨69708⟩⟩
def transferEvent : Nat := 289333
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289329 .summary, .result 284484 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289329 .summary)
      LeftBound289328.bound (LeftBound289328.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69707⟩⟩) (rawTerms := some (Proof.Events1130.exact289329RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289328.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 284484 .summary)
      LeftBound284483.bound (LeftBound284483.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28142⟩⟩) (rawTerms := some (Proof.Events1111.exact284484RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound284483.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289328.bound, LeftBound284483.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289328.bound, LeftBound284483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289328.actual selector witness, LeftBound284483.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289333

namespace LeftBound289337
def owner : Owner := ⟨.program ⟨257⟩, ⟨69709⟩⟩
def transferEvent : Nat := 289337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289335 .coefficient, .predecessor 1 289336 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289335 .coefficient)
      LeftBound289332.bound (LeftBound289332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289332.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289336 .coefficient)
      LeftBound284000.bound (LeftBound284000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1109.exact284004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound284000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound284000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289332.bound, LeftBound284000.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289332.bound, LeftBound284000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289332.actual selector witness, LeftBound284000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289337

namespace LeftBound289338
def owner : Owner := ⟨.program ⟨257⟩, ⟨69709⟩⟩
def transferEvent : Nat := 289338
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289334 .summary, .result 284004 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289334 .summary)
      LeftBound289333.bound (LeftBound289333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69708⟩⟩) (rawTerms := some (Proof.Events1130.exact289334RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 284004 .summary)
      LeftBound284003.bound (LeftBound284003.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30822⟩⟩) (rawTerms := some (Proof.Events1109.exact284004RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound284003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289333.bound, LeftBound284003.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289333.bound, LeftBound284003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289333.actual selector witness, LeftBound284003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289338

namespace LeftBound289342
def owner : Owner := ⟨.program ⟨257⟩, ⟨69710⟩⟩
def transferEvent : Nat := 289342
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289340 .coefficient, .predecessor 1 289341 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289340 .coefficient)
      LeftBound289337.bound (LeftBound289337.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289339RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289337.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289341 .coefficient)
      LeftBound283520.bound (LeftBound283520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1107.exact283524RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283520.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289337.bound, LeftBound283520.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289337.bound, LeftBound283520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289337.actual selector witness, LeftBound283520.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289342

namespace LeftBound289343
def owner : Owner := ⟨.program ⟨257⟩, ⟨69710⟩⟩
def transferEvent : Nat := 289343
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289339 .summary, .result 283524 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289339 .summary)
      LeftBound289338.bound (LeftBound289338.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69709⟩⟩) (rawTerms := some (Proof.Events1130.exact289339RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289338.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283524 .summary)
      LeftBound283523.bound (LeftBound283523.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36482⟩⟩) (rawTerms := some (Proof.Events1107.exact283524RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound283523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289338.bound, LeftBound283523.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289338.bound, LeftBound283523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289338.actual selector witness, LeftBound283523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289343

namespace LeftBound289347
def owner : Owner := ⟨.program ⟨257⟩, ⟨69711⟩⟩
def transferEvent : Nat := 289347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289345 .coefficient, .predecessor 1 289346 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289345 .coefficient)
      LeftBound289342.bound (LeftBound289342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289344RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289346 .coefficient)
      LeftBound283040.bound (LeftBound283040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1105.exact283044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound283040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound283040.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289342.bound, LeftBound283040.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289342.bound, LeftBound283040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289342.actual selector witness, LeftBound283040.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289347

namespace LeftBound289348
def owner : Owner := ⟨.program ⟨257⟩, ⟨69711⟩⟩
def transferEvent : Nat := 289348
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289344 .summary, .result 283044 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289344 .summary)
      LeftBound289343.bound (LeftBound289343.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69710⟩⟩) (rawTerms := some (Proof.Events1130.exact289344RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 283044 .summary)
      LeftBound283043.bound (LeftBound283043.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39162⟩⟩) (rawTerms := some (Proof.Events1105.exact283044RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound283043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289343.bound, LeftBound283043.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289343.bound, LeftBound283043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289343.actual selector witness, LeftBound283043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289348

namespace LeftBound289352
def owner : Owner := ⟨.program ⟨257⟩, ⟨69712⟩⟩
def transferEvent : Nat := 289352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289350 .coefficient, .predecessor 1 289351 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289350 .coefficient)
      LeftBound289347.bound (LeftBound289347.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289347.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289351 .coefficient)
      LeftBound282560.bound (LeftBound282560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1103.exact282564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289347.bound, LeftBound282560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289347.bound, LeftBound282560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289347.actual selector witness, LeftBound282560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289352

namespace LeftBound289353
def owner : Owner := ⟨.program ⟨257⟩, ⟨69712⟩⟩
def transferEvent : Nat := 289353
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289349 .summary, .result 282564 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289349 .summary)
      LeftBound289348.bound (LeftBound289348.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69711⟩⟩) (rawTerms := some (Proof.Events1130.exact289349RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289348.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 282564 .summary)
      LeftBound282563.bound (LeftBound282563.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41842⟩⟩) (rawTerms := some (Proof.Events1103.exact282564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound282563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289348.bound, LeftBound282563.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289348.bound, LeftBound282563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289348.actual selector witness, LeftBound282563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289353

namespace LeftBound289357
def owner : Owner := ⟨.program ⟨257⟩, ⟨69713⟩⟩
def transferEvent : Nat := 289357
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289355 .coefficient, .predecessor 1 289356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289355 .coefficient)
      LeftBound289352.bound (LeftBound289352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289354RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289356 .coefficient)
      LeftBound282080.bound (LeftBound282080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1101.exact282084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound282080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound282080.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289352.bound, LeftBound282080.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289352.bound, LeftBound282080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289352.actual selector witness, LeftBound282080.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289357

namespace LeftBound289358
def owner : Owner := ⟨.program ⟨257⟩, ⟨69713⟩⟩
def transferEvent : Nat := 289358
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289354 .summary, .result 282084 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289354 .summary)
      LeftBound289353.bound (LeftBound289353.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69712⟩⟩) (rawTerms := some (Proof.Events1130.exact289354RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289353.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 282084 .summary)
      LeftBound282083.bound (LeftBound282083.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44522⟩⟩) (rawTerms := some (Proof.Events1101.exact282084RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound282083.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289353.bound, LeftBound282083.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289353.bound, LeftBound282083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289353.actual selector witness, LeftBound282083.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289358

namespace LeftBound289362
def owner : Owner := ⟨.program ⟨257⟩, ⟨69714⟩⟩
def transferEvent : Nat := 289362
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289360 .coefficient, .predecessor 1 289361 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289360 .coefficient)
      LeftBound289357.bound (LeftBound289357.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289359RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289357.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289361 .coefficient)
      LeftBound281600.bound (LeftBound281600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1100.exact281604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound281600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound281600.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289357.bound, LeftBound281600.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289357.bound, LeftBound281600.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289357.actual selector witness, LeftBound281600.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289362

namespace LeftBound289363
def owner : Owner := ⟨.program ⟨257⟩, ⟨69714⟩⟩
def transferEvent : Nat := 289363
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289359 .summary, .result 281604 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289359 .summary)
      LeftBound289358.bound (LeftBound289358.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69713⟩⟩) (rawTerms := some (Proof.Events1130.exact289359RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289358.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 281604 .summary)
      LeftBound281603.bound (LeftBound281603.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47202⟩⟩) (rawTerms := some (Proof.Events1100.exact281604RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound281603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289358.bound, LeftBound281603.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289358.bound, LeftBound281603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289358.actual selector witness, LeftBound281603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289363

namespace LeftBound289367
def owner : Owner := ⟨.program ⟨257⟩, ⟨69715⟩⟩
def transferEvent : Nat := 289367
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 289365 .coefficient, .predecessor 1 289366 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289365 .coefficient)
      LeftBound289362.bound (LeftBound289362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289366 .coefficient)
      LeftBound281120.bound (LeftBound281120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1098.exact281124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound281120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound281120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289362.bound, LeftBound281120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289362.bound, LeftBound281120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289362.actual selector witness, LeftBound281120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289367

namespace LeftBound289368
def owner : Owner := ⟨.program ⟨257⟩, ⟨69715⟩⟩
def transferEvent : Nat := 289368
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 289364 .summary, .result 281124 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 289364 .summary)
      LeftBound289363.bound (LeftBound289363.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69714⟩⟩) (rawTerms := some (Proof.Events1130.exact289364RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound289363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 281124 .summary)
      LeftBound281123.bound (LeftBound281123.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49882⟩⟩) (rawTerms := some (Proof.Events1098.exact281124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound281123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound289363.bound, LeftBound281123.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289363.bound, LeftBound281123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound289363.actual selector witness, LeftBound281123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound289368

namespace LeftBound289372
def owner : Owner := ⟨.program ⟨257⟩, ⟨71050⟩⟩
def transferEvent : Nat := 289372
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 289370 .coefficient) (.predecessor 1 289371 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 289370 .coefficient)
      LeftBound289367.bound (LeftBound289367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1130.exact289369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound289367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound289367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 289371 .coefficient)
      LeftAuthority280627.bound (LeftAuthority280627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority280627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority280627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound289367.bound LeftAuthority280627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound289367.bound, LeftAuthority280627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound289367.actual selector witness) * (LeftAuthority280627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound289372

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
