import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard102
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard106
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard110
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard114
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard125
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard126
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard129
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard137

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26075
def owner : Owner := ⟨.program ⟨257⟩, ⟨23606⟩⟩
def transferEvent : Nat := 26075
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26073 .coefficient, .predecessor 1 26074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26073 .coefficient)
      LeftBound26070.bound (LeftBound26070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26070.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26074 .coefficient)
      LeftBound25061.bound (LeftBound25061.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events097.exact25065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25061.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25061.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26070.bound, LeftBound25061.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26070.bound, LeftBound25061.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26070.actual selector witness, LeftBound25061.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26075

namespace LeftBound26076
def owner : Owner := ⟨.program ⟨257⟩, ⟨23606⟩⟩
def transferEvent : Nat := 26076
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26072 .summary, .result 25065 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26072 .summary)
      LeftBound26071.bound (LeftBound26071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20386⟩⟩) (rawTerms := some (Proof.Events101.exact26072RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25065 .summary)
      LeftBound25064.bound (LeftBound25064.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23605⟩⟩) (rawTerms := some (Proof.Events097.exact25065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25064.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26071.bound, LeftBound25064.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26071.bound, LeftBound25064.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26071.actual selector witness, LeftBound25064.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26076

namespace LeftBound26080
def owner : Owner := ⟨.program ⟨257⟩, ⟨33626⟩⟩
def transferEvent : Nat := 26080
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26078 .coefficient, .predecessor 1 26079 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26078 .coefficient)
      LeftBound26075.bound (LeftBound26075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26079 .coefficient)
      LeftBound24560.bound (LeftBound24560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events095.exact24564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26075.bound, LeftBound24560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26075.bound, LeftBound24560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26075.actual selector witness, LeftBound24560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26080

namespace LeftBound26081
def owner : Owner := ⟨.program ⟨257⟩, ⟨33626⟩⟩
def transferEvent : Nat := 26081
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26077 .summary, .result 24564 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26077 .summary)
      LeftBound26076.bound (LeftBound26076.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23606⟩⟩) (rawTerms := some (Proof.Events101.exact26077RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24564 .summary)
      LeftBound24563.bound (LeftBound24563.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33625⟩⟩) (rawTerms := some (Proof.Events095.exact24564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26076.bound, LeftBound24563.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26076.bound, LeftBound24563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26076.actual selector witness, LeftBound24563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26081

namespace LeftBound26085
def owner : Owner := ⟨.program ⟨257⟩, ⟨52686⟩⟩
def transferEvent : Nat := 26085
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26083 .coefficient, .predecessor 1 26084 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26083 .coefficient)
      LeftBound26080.bound (LeftBound26080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26080.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26084 .coefficient)
      LeftBound24059.bound (LeftBound24059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26080.bound, LeftBound24059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26080.bound, LeftBound24059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26080.actual selector witness, LeftBound24059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26085

namespace LeftBound26086
def owner : Owner := ⟨.program ⟨257⟩, ⟨52686⟩⟩
def transferEvent : Nat := 26086
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26082 .summary, .result 24063 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26082 .summary)
      LeftBound26081.bound (LeftBound26081.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33626⟩⟩) (rawTerms := some (Proof.Events101.exact26082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24063 .summary)
      LeftBound24062.bound (LeftBound24062.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52685⟩⟩) (rawTerms := some (Proof.Events093.exact24063RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26081.bound, LeftBound24062.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26081.bound, LeftBound24062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26081.actual selector witness, LeftBound24062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26086

namespace LeftBound26090
def owner : Owner := ⟨.program ⟨257⟩, ⟨55666⟩⟩
def transferEvent : Nat := 26090
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26088 .coefficient, .predecessor 1 26089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26088 .coefficient)
      LeftBound26085.bound (LeftBound26085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26089 .coefficient)
      LeftBound23558.bound (LeftBound23558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26085.bound, LeftBound23558.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26085.bound, LeftBound23558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26085.actual selector witness, LeftBound23558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26090

namespace LeftBound26091
def owner : Owner := ⟨.program ⟨257⟩, ⟨55666⟩⟩
def transferEvent : Nat := 26091
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26087 .summary, .result 23562 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26087 .summary)
      LeftBound26086.bound (LeftBound26086.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52686⟩⟩) (rawTerms := some (Proof.Events101.exact26087RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23562 .summary)
      LeftBound23561.bound (LeftBound23561.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55665⟩⟩) (rawTerms := some (Proof.Events092.exact23562RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26086.bound, LeftBound23561.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26086.bound, LeftBound23561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26086.actual selector witness, LeftBound23561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26091

namespace LeftBound26095
def owner : Owner := ⟨.program ⟨257⟩, ⟨58646⟩⟩
def transferEvent : Nat := 26095
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26093 .coefficient, .predecessor 1 26094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26093 .coefficient)
      LeftBound26090.bound (LeftBound26090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26094 .coefficient)
      LeftBound23057.bound (LeftBound23057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26090.bound, LeftBound23057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26090.bound, LeftBound23057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26090.actual selector witness, LeftBound23057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26095

namespace LeftBound26096
def owner : Owner := ⟨.program ⟨257⟩, ⟨58646⟩⟩
def transferEvent : Nat := 26096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26092 .summary, .result 23061 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26092 .summary)
      LeftBound26091.bound (LeftBound26091.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55666⟩⟩) (rawTerms := some (Proof.Events101.exact26092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23061 .summary)
      LeftBound23060.bound (LeftBound23060.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58645⟩⟩) (rawTerms := some (Proof.Events090.exact23061RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26091.bound, LeftBound23060.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26091.bound, LeftBound23060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26091.actual selector witness, LeftBound23060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26096

namespace LeftBound26100
def owner : Owner := ⟨.program ⟨257⟩, ⟨61626⟩⟩
def transferEvent : Nat := 26100
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26098 .coefficient, .predecessor 1 26099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26098 .coefficient)
      LeftBound26095.bound (LeftBound26095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26099 .coefficient)
      LeftBound22556.bound (LeftBound22556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events088.exact22560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22556.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26095.bound, LeftBound22556.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26095.bound, LeftBound22556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26095.actual selector witness, LeftBound22556.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26100

namespace LeftBound26101
def owner : Owner := ⟨.program ⟨257⟩, ⟨61626⟩⟩
def transferEvent : Nat := 26101
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26097 .summary, .result 22560 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26097 .summary)
      LeftBound26096.bound (LeftBound26096.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58646⟩⟩) (rawTerms := some (Proof.Events101.exact26097RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22560 .summary)
      LeftBound22559.bound (LeftBound22559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61625⟩⟩) (rawTerms := some (Proof.Events088.exact22560RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26096.bound, LeftBound22559.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26096.bound, LeftBound22559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26096.actual selector witness, LeftBound22559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26101

namespace LeftBound26105
def owner : Owner := ⟨.program ⟨257⟩, ⟨64606⟩⟩
def transferEvent : Nat := 26105
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26103 .coefficient, .predecessor 1 26104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26103 .coefficient)
      LeftBound26100.bound (LeftBound26100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26104 .coefficient)
      LeftBound22055.bound (LeftBound22055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22055.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26100.bound, LeftBound22055.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26100.bound, LeftBound22055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26100.actual selector witness, LeftBound22055.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26105

namespace LeftBound26106
def owner : Owner := ⟨.program ⟨257⟩, ⟨64606⟩⟩
def transferEvent : Nat := 26106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26102 .summary, .result 22059 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26102 .summary)
      LeftBound26101.bound (LeftBound26101.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61626⟩⟩) (rawTerms := some (Proof.Events101.exact26102RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22059 .summary)
      LeftBound22058.bound (LeftBound22058.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64605⟩⟩) (rawTerms := some (Proof.Events086.exact22059RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26101.bound, LeftBound22058.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26101.bound, LeftBound22058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26101.actual selector witness, LeftBound22058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26106

namespace LeftBound26110
def owner : Owner := ⟨.program ⟨257⟩, ⟨69495⟩⟩
def transferEvent : Nat := 26110
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26108 .coefficient, .predecessor 1 26109 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26108 .coefficient)
      LeftBound26105.bound (LeftBound26105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events101.exact26107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26109 .coefficient)
      LeftBound21554.bound (LeftBound21554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26105.bound, LeftBound21554.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26105.bound, LeftBound21554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26105.actual selector witness, LeftBound21554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26110

namespace LeftBound26111
def owner : Owner := ⟨.program ⟨257⟩, ⟨69495⟩⟩
def transferEvent : Nat := 26111
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26107 .summary, .result 21558 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26107 .summary)
      LeftBound26106.bound (LeftBound26106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64606⟩⟩) (rawTerms := some (Proof.Events101.exact26107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21558 .summary)
      LeftBound21557.bound (LeftBound21557.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69494⟩⟩) (rawTerms := some (Proof.Events084.exact21558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26106.bound, LeftBound21557.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26106.bound, LeftBound21557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26106.actual selector witness, LeftBound21557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26111

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
