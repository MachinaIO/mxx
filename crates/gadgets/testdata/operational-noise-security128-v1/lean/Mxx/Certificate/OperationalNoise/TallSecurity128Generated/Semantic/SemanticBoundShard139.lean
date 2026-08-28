import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard082
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard094
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard138

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26115
def owner : Owner := ⟨.program ⟨257⟩, ⟨69496⟩⟩
def transferEvent : Nat := 26115
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26113 .coefficient, .predecessor 1 26114 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26113 .coefficient)
      LeftBound26110.bound (LeftBound26110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26112RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26114 .coefficient)
      LeftBound21053.bound (LeftBound21053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21053.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26110.bound, LeftBound21053.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26110.bound, LeftBound21053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26110.actual selector witness, LeftBound21053.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26115

namespace LeftBound26116
def owner : Owner := ⟨.program ⟨257⟩, ⟨69496⟩⟩
def transferEvent : Nat := 26116
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26112 .summary, .result 21057 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26112 .summary)
      LeftBound26111.bound (LeftBound26111.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69495⟩⟩) (rawTerms := some (Proof.Events102.exact26112RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21057 .summary)
      LeftBound21056.bound (LeftBound21056.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28074⟩⟩) (rawTerms := some (Proof.Events082.exact21057RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21056.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26111.bound, LeftBound21056.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26111.bound, LeftBound21056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26111.actual selector witness, LeftBound21056.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26116

namespace LeftBound26120
def owner : Owner := ⟨.program ⟨257⟩, ⟨69497⟩⟩
def transferEvent : Nat := 26120
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26118 .coefficient, .predecessor 1 26119 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26118 .coefficient)
      LeftBound26115.bound (LeftBound26115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26115.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26119 .coefficient)
      LeftBound20552.bound (LeftBound20552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events080.exact20556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26115.bound, LeftBound20552.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26115.bound, LeftBound20552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26115.actual selector witness, LeftBound20552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26120

namespace LeftBound26121
def owner : Owner := ⟨.program ⟨257⟩, ⟨69497⟩⟩
def transferEvent : Nat := 26121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26117 .summary, .result 20556 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26117 .summary)
      LeftBound26116.bound (LeftBound26116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69496⟩⟩) (rawTerms := some (Proof.Events102.exact26117RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20556 .summary)
      LeftBound20555.bound (LeftBound20555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30754⟩⟩) (rawTerms := some (Proof.Events080.exact20556RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20555.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26116.bound, LeftBound20555.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26116.bound, LeftBound20555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26116.actual selector witness, LeftBound20555.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26121

namespace LeftBound26125
def owner : Owner := ⟨.program ⟨257⟩, ⟨69498⟩⟩
def transferEvent : Nat := 26125
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26123 .coefficient, .predecessor 1 26124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26123 .coefficient)
      LeftBound26120.bound (LeftBound26120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26124 .coefficient)
      LeftBound20051.bound (LeftBound20051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20051.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26120.bound, LeftBound20051.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26120.bound, LeftBound20051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26120.actual selector witness, LeftBound20051.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26125

namespace LeftBound26126
def owner : Owner := ⟨.program ⟨257⟩, ⟨69498⟩⟩
def transferEvent : Nat := 26126
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26122 .summary, .result 20055 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26122 .summary)
      LeftBound26121.bound (LeftBound26121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69497⟩⟩) (rawTerms := some (Proof.Events102.exact26122RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20055 .summary)
      LeftBound20054.bound (LeftBound20054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36414⟩⟩) (rawTerms := some (Proof.Events078.exact20055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26121.bound, LeftBound20054.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26121.bound, LeftBound20054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26121.actual selector witness, LeftBound20054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26126

namespace LeftBound26130
def owner : Owner := ⟨.program ⟨257⟩, ⟨69499⟩⟩
def transferEvent : Nat := 26130
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26128 .coefficient, .predecessor 1 26129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26128 .coefficient)
      LeftBound26125.bound (LeftBound26125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26129 .coefficient)
      LeftBound19550.bound (LeftBound19550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events076.exact19554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26125.bound, LeftBound19550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26125.bound, LeftBound19550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26125.actual selector witness, LeftBound19550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26130

namespace LeftBound26131
def owner : Owner := ⟨.program ⟨257⟩, ⟨69499⟩⟩
def transferEvent : Nat := 26131
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26127 .summary, .result 19554 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26127 .summary)
      LeftBound26126.bound (LeftBound26126.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69498⟩⟩) (rawTerms := some (Proof.Events102.exact26127RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19554 .summary)
      LeftBound19553.bound (LeftBound19553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39094⟩⟩) (rawTerms := some (Proof.Events076.exact19554RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26126.bound, LeftBound19553.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26126.bound, LeftBound19553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26126.actual selector witness, LeftBound19553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26131

namespace LeftBound26135
def owner : Owner := ⟨.program ⟨257⟩, ⟨69500⟩⟩
def transferEvent : Nat := 26135
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26133 .coefficient, .predecessor 1 26134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26133 .coefficient)
      LeftBound26130.bound (LeftBound26130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26134 .coefficient)
      LeftBound19049.bound (LeftBound19049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26130.bound, LeftBound19049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26130.bound, LeftBound19049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26130.actual selector witness, LeftBound19049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26135

namespace LeftBound26136
def owner : Owner := ⟨.program ⟨257⟩, ⟨69500⟩⟩
def transferEvent : Nat := 26136
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26132 .summary, .result 19053 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26132 .summary)
      LeftBound26131.bound (LeftBound26131.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69499⟩⟩) (rawTerms := some (Proof.Events102.exact26132RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26131.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 19053 .summary)
      LeftBound19052.bound (LeftBound19052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41774⟩⟩) (rawTerms := some (Proof.Events074.exact19053RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26131.bound, LeftBound19052.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26131.bound, LeftBound19052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26131.actual selector witness, LeftBound19052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26136

namespace LeftBound26140
def owner : Owner := ⟨.program ⟨257⟩, ⟨69501⟩⟩
def transferEvent : Nat := 26140
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26138 .coefficient, .predecessor 1 26139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26138 .coefficient)
      LeftBound26135.bound (LeftBound26135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26139 .coefficient)
      LeftBound18548.bound (LeftBound18548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26135.bound, LeftBound18548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26135.bound, LeftBound18548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26135.actual selector witness, LeftBound18548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26140

namespace LeftBound26141
def owner : Owner := ⟨.program ⟨257⟩, ⟨69501⟩⟩
def transferEvent : Nat := 26141
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26137 .summary, .result 18552 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26137 .summary)
      LeftBound26136.bound (LeftBound26136.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69500⟩⟩) (rawTerms := some (Proof.Events102.exact26137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18552 .summary)
      LeftBound18551.bound (LeftBound18551.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44454⟩⟩) (rawTerms := some (Proof.Events072.exact18552RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26136.bound, LeftBound18551.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26136.bound, LeftBound18551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26136.actual selector witness, LeftBound18551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26141

namespace LeftBound26145
def owner : Owner := ⟨.program ⟨257⟩, ⟨69502⟩⟩
def transferEvent : Nat := 26145
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26143 .coefficient, .predecessor 1 26144 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26143 .coefficient)
      LeftBound26140.bound (LeftBound26140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26144 .coefficient)
      LeftBound18047.bound (LeftBound18047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18047.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26140.bound, LeftBound18047.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26140.bound, LeftBound18047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26140.actual selector witness, LeftBound18047.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26145

namespace LeftBound26146
def owner : Owner := ⟨.program ⟨257⟩, ⟨69502⟩⟩
def transferEvent : Nat := 26146
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26142 .summary, .result 18051 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26142 .summary)
      LeftBound26141.bound (LeftBound26141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69501⟩⟩) (rawTerms := some (Proof.Events102.exact26142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 18051 .summary)
      LeftBound18050.bound (LeftBound18050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47134⟩⟩) (rawTerms := some (Proof.Events070.exact18051RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18050.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26141.bound, LeftBound18050.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26141.bound, LeftBound18050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26141.actual selector witness, LeftBound18050.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26146

namespace LeftBound26150
def owner : Owner := ⟨.program ⟨257⟩, ⟨69503⟩⟩
def transferEvent : Nat := 26150
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 26148 .coefficient, .predecessor 1 26149 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26148 .coefficient)
      LeftBound26145.bound (LeftBound26145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26149 .coefficient)
      LeftBound17546.bound (LeftBound17546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17546.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26145.bound, LeftBound17546.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26145.bound, LeftBound17546.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26145.actual selector witness, LeftBound17546.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26150

namespace LeftBound26151
def owner : Owner := ⟨.program ⟨257⟩, ⟨69503⟩⟩
def transferEvent : Nat := 26151
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 26147 .summary, .result 17550 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26147 .summary)
      LeftBound26146.bound (LeftBound26146.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69502⟩⟩) (rawTerms := some (Proof.Events102.exact26147RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26146.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17550 .summary)
      LeftBound17549.bound (LeftBound17549.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49814⟩⟩) (rawTerms := some (Proof.Events068.exact17550RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26146.bound, LeftBound17549.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26146.bound, LeftBound17549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound26146.actual selector witness, LeftBound17549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26151

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
