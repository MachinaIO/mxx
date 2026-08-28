import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard902
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard905
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard909
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard913
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard916
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard920
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard923
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard924
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard927
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard931
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard949

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound143094
def owner : Owner := ⟨.program ⟨257⟩, ⟨55719⟩⟩
def transferEvent : Nat := 143094
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143090 .summary, .result 140660 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143090 .summary)
      LeftBound143089.bound (LeftBound143089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52739⟩⟩) (rawTerms := some (Proof.Events558.exact143090RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140660 .summary)
      LeftBound140659.bound (LeftBound140659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55718⟩⟩) (rawTerms := some (Proof.Events549.exact140660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound140659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143089.bound, LeftBound140659.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143089.bound, LeftBound140659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143089.actual selector witness, LeftBound140659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143094

namespace LeftBound143098
def owner : Owner := ⟨.program ⟨257⟩, ⟨58699⟩⟩
def transferEvent : Nat := 143098
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143096 .coefficient, .predecessor 1 143097 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143096 .coefficient)
      LeftBound143093.bound (LeftBound143093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143097 .coefficient)
      LeftBound140174.bound (LeftBound140174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events547.exact140178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143093.bound, LeftBound140174.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143093.bound, LeftBound140174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143093.actual selector witness, LeftBound140174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143098

namespace LeftBound143099
def owner : Owner := ⟨.program ⟨257⟩, ⟨58699⟩⟩
def transferEvent : Nat := 143099
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143095 .summary, .result 140178 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143095 .summary)
      LeftBound143094.bound (LeftBound143094.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55719⟩⟩) (rawTerms := some (Proof.Events558.exact143095RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 140178 .summary)
      LeftBound140177.bound (LeftBound140177.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58698⟩⟩) (rawTerms := some (Proof.Events547.exact140178RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound140177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143094.bound, LeftBound140177.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143094.bound, LeftBound140177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143094.actual selector witness, LeftBound140177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143099

namespace LeftBound143103
def owner : Owner := ⟨.program ⟨257⟩, ⟨61679⟩⟩
def transferEvent : Nat := 143103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143101 .coefficient, .predecessor 1 143102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143101 .coefficient)
      LeftBound143098.bound (LeftBound143098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143098.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143098.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143102 .coefficient)
      LeftBound139692.bound (LeftBound139692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events545.exact139696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143098.bound, LeftBound139692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143098.bound, LeftBound139692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143098.actual selector witness, LeftBound139692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143103

namespace LeftBound143104
def owner : Owner := ⟨.program ⟨257⟩, ⟨61679⟩⟩
def transferEvent : Nat := 143104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143100 .summary, .result 139696 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143100 .summary)
      LeftBound143099.bound (LeftBound143099.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58699⟩⟩) (rawTerms := some (Proof.Events558.exact143100RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 139696 .summary)
      LeftBound139695.bound (LeftBound139695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61678⟩⟩) (rawTerms := some (Proof.Events545.exact139696RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound139695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143099.bound, LeftBound139695.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143099.bound, LeftBound139695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143099.actual selector witness, LeftBound139695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143104

namespace LeftBound143108
def owner : Owner := ⟨.program ⟨257⟩, ⟨64659⟩⟩
def transferEvent : Nat := 143108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143106 .coefficient, .predecessor 1 143107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143106 .coefficient)
      LeftBound143103.bound (LeftBound143103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143107 .coefficient)
      LeftBound139210.bound (LeftBound139210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events543.exact139214RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound139210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound139210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143103.bound, LeftBound139210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143103.bound, LeftBound139210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143103.actual selector witness, LeftBound139210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143108

namespace LeftBound143109
def owner : Owner := ⟨.program ⟨257⟩, ⟨64659⟩⟩
def transferEvent : Nat := 143109
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143105 .summary, .result 139214 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143105 .summary)
      LeftBound143104.bound (LeftBound143104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61679⟩⟩) (rawTerms := some (Proof.Events559.exact143105RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 139214 .summary)
      LeftBound139213.bound (LeftBound139213.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64658⟩⟩) (rawTerms := some (Proof.Events543.exact139214RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound139213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143104.bound, LeftBound139213.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143104.bound, LeftBound139213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143104.actual selector witness, LeftBound139213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143109

namespace LeftBound143113
def owner : Owner := ⟨.program ⟨257⟩, ⟨69628⟩⟩
def transferEvent : Nat := 143113
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143111 .coefficient, .predecessor 1 143112 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143111 .coefficient)
      LeftBound143108.bound (LeftBound143108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143112 .coefficient)
      LeftBound138728.bound (LeftBound138728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events541.exact138732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143108.bound, LeftBound138728.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143108.bound, LeftBound138728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143108.actual selector witness, LeftBound138728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143113

namespace LeftBound143114
def owner : Owner := ⟨.program ⟨257⟩, ⟨69628⟩⟩
def transferEvent : Nat := 143114
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143110 .summary, .result 138732 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143110 .summary)
      LeftBound143109.bound (LeftBound143109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64659⟩⟩) (rawTerms := some (Proof.Events559.exact143110RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138732 .summary)
      LeftBound138731.bound (LeftBound138731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69627⟩⟩) (rawTerms := some (Proof.Events541.exact138732RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound138731.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143109.bound, LeftBound138731.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143109.bound, LeftBound138731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143109.actual selector witness, LeftBound138731.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143114

namespace LeftBound143118
def owner : Owner := ⟨.program ⟨257⟩, ⟨69629⟩⟩
def transferEvent : Nat := 143118
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143116 .coefficient, .predecessor 1 143117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143116 .coefficient)
      LeftBound143113.bound (LeftBound143113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143117 .coefficient)
      LeftBound138246.bound (LeftBound138246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events540.exact138250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound138246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound138246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143113.bound, LeftBound138246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143113.bound, LeftBound138246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143113.actual selector witness, LeftBound138246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143118

namespace LeftBound143119
def owner : Owner := ⟨.program ⟨257⟩, ⟨69629⟩⟩
def transferEvent : Nat := 143119
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143115 .summary, .result 138250 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143115 .summary)
      LeftBound143114.bound (LeftBound143114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69628⟩⟩) (rawTerms := some (Proof.Events559.exact143115RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 138250 .summary)
      LeftBound138249.bound (LeftBound138249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28117⟩⟩) (rawTerms := some (Proof.Events540.exact138250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound138249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143114.bound, LeftBound138249.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143114.bound, LeftBound138249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143114.actual selector witness, LeftBound138249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143119

namespace LeftBound143123
def owner : Owner := ⟨.program ⟨257⟩, ⟨69630⟩⟩
def transferEvent : Nat := 143123
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143121 .coefficient, .predecessor 1 143122 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143121 .coefficient)
      LeftBound143118.bound (LeftBound143118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143118.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143122 .coefficient)
      LeftBound137764.bound (LeftBound137764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events538.exact137768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143118.bound, LeftBound137764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143118.bound, LeftBound137764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143118.actual selector witness, LeftBound137764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143123

namespace LeftBound143124
def owner : Owner := ⟨.program ⟨257⟩, ⟨69630⟩⟩
def transferEvent : Nat := 143124
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143120 .summary, .result 137768 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143120 .summary)
      LeftBound143119.bound (LeftBound143119.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69629⟩⟩) (rawTerms := some (Proof.Events559.exact143120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 137768 .summary)
      LeftBound137767.bound (LeftBound137767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30797⟩⟩) (rawTerms := some (Proof.Events538.exact137768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound137767.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143119.bound, LeftBound137767.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143119.bound, LeftBound137767.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143119.actual selector witness, LeftBound137767.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143124

namespace LeftBound143128
def owner : Owner := ⟨.program ⟨257⟩, ⟨69631⟩⟩
def transferEvent : Nat := 143128
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143126 .coefficient, .predecessor 1 143127 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143126 .coefficient)
      LeftBound143123.bound (LeftBound143123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143123.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143127 .coefficient)
      LeftBound137282.bound (LeftBound137282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events536.exact137286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound137282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound137282.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143123.bound, LeftBound137282.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143123.bound, LeftBound137282.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143123.actual selector witness, LeftBound137282.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143128

namespace LeftBound143129
def owner : Owner := ⟨.program ⟨257⟩, ⟨69631⟩⟩
def transferEvent : Nat := 143129
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143125 .summary, .result 137286 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143125 .summary)
      LeftBound143124.bound (LeftBound143124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69630⟩⟩) (rawTerms := some (Proof.Events559.exact143125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143124.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 137286 .summary)
      LeftBound137285.bound (LeftBound137285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36457⟩⟩) (rawTerms := some (Proof.Events536.exact137286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound137285.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143124.bound, LeftBound137285.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143124.bound, LeftBound137285.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143124.actual selector witness, LeftBound137285.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143129

namespace LeftBound143133
def owner : Owner := ⟨.program ⟨257⟩, ⟨69632⟩⟩
def transferEvent : Nat := 143133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143131 .coefficient, .predecessor 1 143132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143131 .coefficient)
      LeftBound143128.bound (LeftBound143128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events559.exact143130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143128.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143132 .coefficient)
      LeftBound136800.bound (LeftBound136800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events534.exact136804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound136800.bound, RecordedBoundRefines] <;> decide)
      (LeftBound136800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143128.bound, LeftBound136800.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143128.bound, LeftBound136800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143128.actual selector witness, LeftBound136800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143133

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
