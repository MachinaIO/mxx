import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard887
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard891
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard894
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard902
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard950

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound143134
def owner : Owner := ⟨.program ⟨257⟩, ⟨69632⟩⟩
def transferEvent : Nat := 143134
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143130 .summary, .result 136804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143130 .summary)
      LeftBound143129.bound (LeftBound143129.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69631⟩⟩) (rawTerms := some (Proof.Events559.exact143130RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 136804 .summary)
      LeftBound136803.bound (LeftBound136803.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39137⟩⟩) (rawTerms := some (Proof.Events534.exact136804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound136803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143129.bound, LeftBound136803.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143129.bound, LeftBound136803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143129.actual selector witness, LeftBound136803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143134

namespace LeftBound143138
def owner : Owner := ⟨.program ⟨257⟩, ⟨69633⟩⟩
def transferEvent : Nat := 143138
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143136 .coefficient, .predecessor 1 143137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143136 .coefficient)
      LeftBound143133.bound (LeftBound143133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143137 .coefficient)
      LeftBound136318.bound (LeftBound136318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events532.exact136322RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound136318.bound, RecordedBoundRefines] <;> decide)
      (LeftBound136318.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143133.bound, LeftBound136318.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143133.bound, LeftBound136318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143133.actual selector witness, LeftBound136318.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143138

namespace LeftBound143139
def owner : Owner := ⟨.program ⟨257⟩, ⟨69633⟩⟩
def transferEvent : Nat := 143139
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143135 .summary, .result 136322 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143135 .summary)
      LeftBound143134.bound (LeftBound143134.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69632⟩⟩) (rawTerms := some (Proof.Events559.exact143135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 136322 .summary)
      LeftBound136321.bound (LeftBound136321.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41817⟩⟩) (rawTerms := some (Proof.Events532.exact136322RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound136321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143134.bound, LeftBound136321.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143134.bound, LeftBound136321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143134.actual selector witness, LeftBound136321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143139

namespace LeftBound143143
def owner : Owner := ⟨.program ⟨257⟩, ⟨69634⟩⟩
def transferEvent : Nat := 143143
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143141 .coefficient, .predecessor 1 143142 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143141 .coefficient)
      LeftBound143138.bound (LeftBound143138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143142 .coefficient)
      LeftBound135836.bound (LeftBound135836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events530.exact135840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143138.bound, LeftBound135836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143138.bound, LeftBound135836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143138.actual selector witness, LeftBound135836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143143

namespace LeftBound143144
def owner : Owner := ⟨.program ⟨257⟩, ⟨69634⟩⟩
def transferEvent : Nat := 143144
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143140 .summary, .result 135840 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143140 .summary)
      LeftBound143139.bound (LeftBound143139.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69633⟩⟩) (rawTerms := some (Proof.Events559.exact143140RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 135840 .summary)
      LeftBound135839.bound (LeftBound135839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44497⟩⟩) (rawTerms := some (Proof.Events530.exact135840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound135839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143139.bound, LeftBound135839.bound]
def bound : CoeffClass := .finite ⟨515053820849391945920019041353728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143139.bound, LeftBound135839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143139.actual selector witness, LeftBound135839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143144

namespace LeftBound143148
def owner : Owner := ⟨.program ⟨257⟩, ⟨69635⟩⟩
def transferEvent : Nat := 143148
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143146 .coefficient, .predecessor 1 143147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143146 .coefficient)
      LeftBound143143.bound (LeftBound143143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143147 .coefficient)
      LeftBound135354.bound (LeftBound135354.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events528.exact135358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135354.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135354.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143143.bound, LeftBound135354.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143143.bound, LeftBound135354.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143143.actual selector witness, LeftBound135354.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143148

namespace LeftBound143149
def owner : Owner := ⟨.program ⟨257⟩, ⟨69635⟩⟩
def transferEvent : Nat := 143149
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143145 .summary, .result 135358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143145 .summary)
      LeftBound143144.bound (LeftBound143144.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69634⟩⟩) (rawTerms := some (Proof.Events559.exact143145RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 135358 .summary)
      LeftBound135357.bound (LeftBound135357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47177⟩⟩) (rawTerms := some (Proof.Events528.exact135358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound135357.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143144.bound, LeftBound135357.bound]
def bound : CoeffClass := .finite ⟨547248128674354899372274579931136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143144.bound, LeftBound135357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143144.actual selector witness, LeftBound135357.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143149

namespace LeftBound143153
def owner : Owner := ⟨.program ⟨257⟩, ⟨69636⟩⟩
def transferEvent : Nat := 143153
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143151 .coefficient, .predecessor 1 143152 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143151 .coefficient)
      LeftBound143148.bound (LeftBound143148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143152 .coefficient)
      LeftBound134872.bound (LeftBound134872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events526.exact134876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143148.bound, LeftBound134872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143148.bound, LeftBound134872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143148.actual selector witness, LeftBound134872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143153

namespace LeftBound143154
def owner : Owner := ⟨.program ⟨257⟩, ⟨69636⟩⟩
def transferEvent : Nat := 143154
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143150 .summary, .result 134876 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143150 .summary)
      LeftBound143149.bound (LeftBound143149.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69635⟩⟩) (rawTerms := some (Proof.Events559.exact143150RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143149.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134876 .summary)
      LeftBound134875.bound (LeftBound134875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49857⟩⟩) (rawTerms := some (Proof.Events526.exact134876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143149.bound, LeftBound134875.bound]
def bound : CoeffClass := .finite ⟨579442632949763540201771008262144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143149.bound, LeftBound134875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143149.actual selector witness, LeftBound134875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143154

namespace LeftBound143158
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def transferEvent : Nat := 143158
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 143156 .coefficient) (.predecessor 1 143157 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143156 .coefficient)
      LeftBound143153.bound (LeftBound143153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143157 .coefficient)
      LeftAuthority134377.bound (LeftAuthority134377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events524.exact134378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority134377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority134377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound143153.bound LeftAuthority134377.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143153.bound, LeftAuthority134377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound143153.actual selector witness) * (LeftAuthority134377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143158

namespace LeftBound143159
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def transferEvent : Nat := 143159
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨71017⟩⟩]⟩ [⟨.result 134378 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134378 .coefficient)
      LeftAuthority134377.bound (LeftAuthority134377.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨71017⟩⟩) (rawTerms := some (Proof.Events524.exact134378RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority134377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority134377.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority134377.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority134377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority134377.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound143159

namespace LeftBound143160
def owner : Owner := ⟨.program ⟨257⟩, ⟨71019⟩⟩
def transferEvent : Nat := 143160
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 143155 .summary) (.transfer 143159) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143155 .summary)
      LeftBound143154.bound (LeftBound143154.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69636⟩⟩) (rawTerms := some (Proof.Events559.exact143155RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 143159)
      LeftBound143159.bound (LeftBound143159.actual selector witness) := by
  exact .transfer (LeftBound143159.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound143154.bound LeftBound143159.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143154.bound, LeftBound143159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound143154.actual selector witness) * (LeftBound143159.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143160

namespace LeftBound143239
def owner : Owner := ⟨.program ⟨257⟩, ⟨68302⟩⟩
def transferEvent : Nat := 143239
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 143237 .coefficient) (.value (.predecessor 1 143238 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143237 .coefficient)
      LeftAuthority143235.bound (LeftAuthority143235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143238 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority143235.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority143235.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority143235.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound143239

namespace LeftBound143243
def owner : Owner := ⟨.program ⟨257⟩, ⟨68303⟩⟩
def transferEvent : Nat := 143243
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 143241 .coefficient) (.predecessor 1 143242 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143241 .coefficient)
      LeftBound134492.bound (LeftBound134492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143242 .coefficient)
      LeftBound143239.bound (LeftBound143239.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143239.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134492.bound LeftBound143239.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134492.bound, LeftBound143239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134492.actual selector witness) * (LeftBound143239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143243

namespace LeftBound143244
def owner : Owner := ⟨.program ⟨257⟩, ⟨68303⟩⟩
def transferEvent : Nat := 143244
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68300⟩⟩]⟩ [⟨.result 143236 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143236 .coefficient)
      LeftAuthority143235.bound (LeftAuthority143235.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68300⟩⟩) (rawTerms := some (Proof.Events559.exact143236RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143235.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority143235.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority143235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority143235.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound143244

namespace LeftBound143245
def owner : Owner := ⟨.program ⟨257⟩, ⟨68303⟩⟩
def transferEvent : Nat := 143245
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 134495 .summary) (.transfer 143244) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134495 .summary)
      LeftBound134493.bound (LeftBound134493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5473⟩⟩) (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 143244)
      LeftBound143244.bound (LeftBound143244.actual selector witness) := by
  exact .transfer (LeftBound143244.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134493.bound LeftBound143244.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134493.bound, LeftBound143244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134493.actual selector witness) * (LeftBound143244.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143245

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
