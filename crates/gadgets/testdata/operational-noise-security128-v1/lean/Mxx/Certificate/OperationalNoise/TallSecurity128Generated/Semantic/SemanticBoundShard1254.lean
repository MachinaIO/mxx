import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1217
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1221
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1224
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1228
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1231
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1235
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1239
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1242
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1253

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound186958
def owner : Owner := ⟨.program ⟨257⟩, ⟨33989⟩⟩
def transferEvent : Nat := 186958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186956 .coefficient, .predecessor 1 186957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186956 .coefficient)
      LeftBound186953.bound (LeftBound186953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186957 .coefficient)
      LeftBound185495.bound (LeftBound185495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events724.exact185499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186953.bound, LeftBound185495.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186953.bound, LeftBound185495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186953.actual selector witness, LeftBound185495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186958

namespace LeftBound186959
def owner : Owner := ⟨.program ⟨257⟩, ⟨33989⟩⟩
def transferEvent : Nat := 186959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186955 .summary, .result 185499 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186955 .summary)
      LeftBound186954.bound (LeftBound186954.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23969⟩⟩) (rawTerms := some (Proof.Events730.exact186955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185499 .summary)
      LeftBound185498.bound (LeftBound185498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33988⟩⟩) (rawTerms := some (Proof.Events724.exact185499RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186954.bound, LeftBound185498.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186954.bound, LeftBound185498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186954.actual selector witness, LeftBound185498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186959

namespace LeftBound186963
def owner : Owner := ⟨.program ⟨257⟩, ⟨53049⟩⟩
def transferEvent : Nat := 186963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186961 .coefficient, .predecessor 1 186962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186961 .coefficient)
      LeftBound186958.bound (LeftBound186958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186962 .coefficient)
      LeftBound185013.bound (LeftBound185013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events722.exact185017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186958.bound, LeftBound185013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186958.bound, LeftBound185013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186958.actual selector witness, LeftBound185013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186963

namespace LeftBound186964
def owner : Owner := ⟨.program ⟨257⟩, ⟨53049⟩⟩
def transferEvent : Nat := 186964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186960 .summary, .result 185017 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186960 .summary)
      LeftBound186959.bound (LeftBound186959.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33989⟩⟩) (rawTerms := some (Proof.Events730.exact186960RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185017 .summary)
      LeftBound185016.bound (LeftBound185016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53048⟩⟩) (rawTerms := some (Proof.Events722.exact185017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186959.bound, LeftBound185016.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186959.bound, LeftBound185016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186959.actual selector witness, LeftBound185016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186964

namespace LeftBound186968
def owner : Owner := ⟨.program ⟨257⟩, ⟨56029⟩⟩
def transferEvent : Nat := 186968
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186966 .coefficient, .predecessor 1 186967 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186966 .coefficient)
      LeftBound186963.bound (LeftBound186963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186967 .coefficient)
      LeftBound184531.bound (LeftBound184531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events720.exact184535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound184531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound184531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186963.bound, LeftBound184531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186963.bound, LeftBound184531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186963.actual selector witness, LeftBound184531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186968

namespace LeftBound186969
def owner : Owner := ⟨.program ⟨257⟩, ⟨56029⟩⟩
def transferEvent : Nat := 186969
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186965 .summary, .result 184535 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186965 .summary)
      LeftBound186964.bound (LeftBound186964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53049⟩⟩) (rawTerms := some (Proof.Events730.exact186965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 184535 .summary)
      LeftBound184534.bound (LeftBound184534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56028⟩⟩) (rawTerms := some (Proof.Events720.exact184535RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound184534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186964.bound, LeftBound184534.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186964.bound, LeftBound184534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186964.actual selector witness, LeftBound184534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186969

namespace LeftBound186973
def owner : Owner := ⟨.program ⟨257⟩, ⟨59009⟩⟩
def transferEvent : Nat := 186973
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186971 .coefficient, .predecessor 1 186972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186971 .coefficient)
      LeftBound186968.bound (LeftBound186968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186972 .coefficient)
      LeftBound184049.bound (LeftBound184049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events718.exact184053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound184049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound184049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186968.bound, LeftBound184049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186968.bound, LeftBound184049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186968.actual selector witness, LeftBound184049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186973

namespace LeftBound186974
def owner : Owner := ⟨.program ⟨257⟩, ⟨59009⟩⟩
def transferEvent : Nat := 186974
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186970 .summary, .result 184053 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186970 .summary)
      LeftBound186969.bound (LeftBound186969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56029⟩⟩) (rawTerms := some (Proof.Events730.exact186970RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 184053 .summary)
      LeftBound184052.bound (LeftBound184052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59008⟩⟩) (rawTerms := some (Proof.Events718.exact184053RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound184052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186969.bound, LeftBound184052.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186969.bound, LeftBound184052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186969.actual selector witness, LeftBound184052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186974

namespace LeftBound186978
def owner : Owner := ⟨.program ⟨257⟩, ⟨61989⟩⟩
def transferEvent : Nat := 186978
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186976 .coefficient, .predecessor 1 186977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186976 .coefficient)
      LeftBound186973.bound (LeftBound186973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186977 .coefficient)
      LeftBound183567.bound (LeftBound183567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events717.exact183571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186973.bound, LeftBound183567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186973.bound, LeftBound183567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186973.actual selector witness, LeftBound183567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186978

namespace LeftBound186979
def owner : Owner := ⟨.program ⟨257⟩, ⟨61989⟩⟩
def transferEvent : Nat := 186979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186975 .summary, .result 183571 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186975 .summary)
      LeftBound186974.bound (LeftBound186974.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59009⟩⟩) (rawTerms := some (Proof.Events730.exact186975RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183571 .summary)
      LeftBound183570.bound (LeftBound183570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61988⟩⟩) (rawTerms := some (Proof.Events717.exact183571RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186974.bound, LeftBound183570.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186974.bound, LeftBound183570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186974.actual selector witness, LeftBound183570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186979

namespace LeftBound186983
def owner : Owner := ⟨.program ⟨257⟩, ⟨64969⟩⟩
def transferEvent : Nat := 186983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186981 .coefficient, .predecessor 1 186982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186981 .coefficient)
      LeftBound186978.bound (LeftBound186978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186982 .coefficient)
      LeftBound183085.bound (LeftBound183085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events715.exact183089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound183085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound183085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186978.bound, LeftBound183085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186978.bound, LeftBound183085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186978.actual selector witness, LeftBound183085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186983

namespace LeftBound186984
def owner : Owner := ⟨.program ⟨257⟩, ⟨64969⟩⟩
def transferEvent : Nat := 186984
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186980 .summary, .result 183089 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186980 .summary)
      LeftBound186979.bound (LeftBound186979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61989⟩⟩) (rawTerms := some (Proof.Events730.exact186980RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 183089 .summary)
      LeftBound183088.bound (LeftBound183088.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64968⟩⟩) (rawTerms := some (Proof.Events715.exact183089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound183088.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186979.bound, LeftBound183088.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186979.bound, LeftBound183088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186979.actual selector witness, LeftBound183088.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186984

namespace LeftBound186988
def owner : Owner := ⟨.program ⟨257⟩, ⟨70418⟩⟩
def transferEvent : Nat := 186988
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186986 .coefficient, .predecessor 1 186987 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186986 .coefficient)
      LeftBound186983.bound (LeftBound186983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186987 .coefficient)
      LeftBound182603.bound (LeftBound182603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events713.exact182607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186983.bound, LeftBound182603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186983.bound, LeftBound182603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186983.actual selector witness, LeftBound182603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186988

namespace LeftBound186989
def owner : Owner := ⟨.program ⟨257⟩, ⟨70418⟩⟩
def transferEvent : Nat := 186989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186985 .summary, .result 182607 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186985 .summary)
      LeftBound186984.bound (LeftBound186984.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64969⟩⟩) (rawTerms := some (Proof.Events730.exact186985RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 182607 .summary)
      LeftBound182606.bound (LeftBound182606.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70417⟩⟩) (rawTerms := some (Proof.Events713.exact182607RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound182606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186984.bound, LeftBound182606.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186984.bound, LeftBound182606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186984.actual selector witness, LeftBound182606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186989

namespace LeftBound186993
def owner : Owner := ⟨.program ⟨257⟩, ⟨70419⟩⟩
def transferEvent : Nat := 186993
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186991 .coefficient, .predecessor 1 186992 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186991 .coefficient)
      LeftBound186988.bound (LeftBound186988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact186990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186992 .coefficient)
      LeftBound182121.bound (LeftBound182121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186988.bound, LeftBound182121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186988.bound, LeftBound182121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186988.actual selector witness, LeftBound182121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186993

namespace LeftBound186994
def owner : Owner := ⟨.program ⟨257⟩, ⟨70419⟩⟩
def transferEvent : Nat := 186994
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 186990 .summary, .result 182125 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186990 .summary)
      LeftBound186989.bound (LeftBound186989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70418⟩⟩) (rawTerms := some (Proof.Events730.exact186990RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 182125 .summary)
      LeftBound182124.bound (LeftBound182124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28367⟩⟩) (rawTerms := some (Proof.Events711.exact182125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound182124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186989.bound, LeftBound182124.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186989.bound, LeftBound182124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186989.actual selector witness, LeftBound182124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186994

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
