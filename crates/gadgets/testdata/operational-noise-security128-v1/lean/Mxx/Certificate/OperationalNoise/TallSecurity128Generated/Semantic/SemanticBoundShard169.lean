import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard143
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard144
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard145
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard146
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard147
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard148
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard149
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard151
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard152
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard168

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound31602
def owner : Owner := ⟨.program ⟨257⟩, ⟨69483⟩⟩
def transferEvent : Nat := 31602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31598 .summary, .result 29181 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31598 .summary)
      LeftBound31597.bound (LeftBound31597.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69482⟩⟩) (rawTerms := some (Proof.Events123.exact31598RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 29181 .summary)
      LeftBound29176.bound (LeftBound29176.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30749⟩⟩) (rawTerms := some (Proof.Events113.exact29181RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31597.bound, LeftBound29176.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31597.bound, LeftBound29176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31597.actual selector witness, LeftBound29176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31602

namespace LeftBound31606
def owner : Owner := ⟨.program ⟨257⟩, ⟨69484⟩⟩
def transferEvent : Nat := 31606
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31604 .coefficient, .predecessor 1 31605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31604 .coefficient)
      LeftBound31601.bound (LeftBound31601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31605 .coefficient)
      LeftBound28962.bound (LeftBound28962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact28969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31601.bound, LeftBound28962.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31601.bound, LeftBound28962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31601.actual selector witness, LeftBound28962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31606

namespace LeftBound31607
def owner : Owner := ⟨.program ⟨257⟩, ⟨69484⟩⟩
def transferEvent : Nat := 31607
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31603 .summary, .result 28969 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31603 .summary)
      LeftBound31602.bound (LeftBound31602.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69483⟩⟩) (rawTerms := some (Proof.Events123.exact31603RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28969 .summary)
      LeftBound28964.bound (LeftBound28964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36409⟩⟩) (rawTerms := some (Proof.Events113.exact28969RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28964.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31602.bound, LeftBound28964.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31602.bound, LeftBound28964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31602.actual selector witness, LeftBound28964.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31607

namespace LeftBound31611
def owner : Owner := ⟨.program ⟨257⟩, ⟨69485⟩⟩
def transferEvent : Nat := 31611
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31609 .coefficient, .predecessor 1 31610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31609 .coefficient)
      LeftBound31606.bound (LeftBound31606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31610 .coefficient)
      LeftBound28750.bound (LeftBound28750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28750.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31606.bound, LeftBound28750.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31606.bound, LeftBound28750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31606.actual selector witness, LeftBound28750.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31611

namespace LeftBound31612
def owner : Owner := ⟨.program ⟨257⟩, ⟨69485⟩⟩
def transferEvent : Nat := 31612
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31608 .summary, .result 28757 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31608 .summary)
      LeftBound31607.bound (LeftBound31607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69484⟩⟩) (rawTerms := some (Proof.Events123.exact31608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28757 .summary)
      LeftBound28752.bound (LeftBound28752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39089⟩⟩) (rawTerms := some (Proof.Events112.exact28757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28752.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31607.bound, LeftBound28752.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31607.bound, LeftBound28752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31607.actual selector witness, LeftBound28752.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31612

namespace LeftBound31616
def owner : Owner := ⟨.program ⟨257⟩, ⟨69486⟩⟩
def transferEvent : Nat := 31616
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31614 .coefficient, .predecessor 1 31615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31614 .coefficient)
      LeftBound31611.bound (LeftBound31611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31615 .coefficient)
      LeftBound28538.bound (LeftBound28538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28538.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31611.bound, LeftBound28538.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31611.bound, LeftBound28538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31611.actual selector witness, LeftBound28538.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31616

namespace LeftBound31617
def owner : Owner := ⟨.program ⟨257⟩, ⟨69486⟩⟩
def transferEvent : Nat := 31617
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31613 .summary, .result 28545 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31613 .summary)
      LeftBound31612.bound (LeftBound31612.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69485⟩⟩) (rawTerms := some (Proof.Events123.exact31613RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28545 .summary)
      LeftBound28540.bound (LeftBound28540.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41769⟩⟩) (rawTerms := some (Proof.Events111.exact28545RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31612.bound, LeftBound28540.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31612.bound, LeftBound28540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31612.actual selector witness, LeftBound28540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31617

namespace LeftBound31621
def owner : Owner := ⟨.program ⟨257⟩, ⟨69487⟩⟩
def transferEvent : Nat := 31621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31619 .coefficient, .predecessor 1 31620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31619 .coefficient)
      LeftBound31616.bound (LeftBound31616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31620 .coefficient)
      LeftBound28326.bound (LeftBound28326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28326.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31616.bound, LeftBound28326.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31616.bound, LeftBound28326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31616.actual selector witness, LeftBound28326.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31621

namespace LeftBound31622
def owner : Owner := ⟨.program ⟨257⟩, ⟨69487⟩⟩
def transferEvent : Nat := 31622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31618 .summary, .result 28333 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31618 .summary)
      LeftBound31617.bound (LeftBound31617.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69486⟩⟩) (rawTerms := some (Proof.Events123.exact31618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28333 .summary)
      LeftBound28328.bound (LeftBound28328.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44449⟩⟩) (rawTerms := some (Proof.Events110.exact28333RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28328.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31617.bound, LeftBound28328.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31617.bound, LeftBound28328.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31617.actual selector witness, LeftBound28328.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31622

namespace LeftBound31626
def owner : Owner := ⟨.program ⟨257⟩, ⟨69488⟩⟩
def transferEvent : Nat := 31626
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31624 .coefficient, .predecessor 1 31625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31624 .coefficient)
      LeftBound31621.bound (LeftBound31621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31625 .coefficient)
      LeftBound28114.bound (LeftBound28114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact28121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28114.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31621.bound, LeftBound28114.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31621.bound, LeftBound28114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31621.actual selector witness, LeftBound28114.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31626

namespace LeftBound31627
def owner : Owner := ⟨.program ⟨257⟩, ⟨69488⟩⟩
def transferEvent : Nat := 31627
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31623 .summary, .result 28121 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31623 .summary)
      LeftBound31622.bound (LeftBound31622.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69487⟩⟩) (rawTerms := some (Proof.Events123.exact31623RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 28121 .summary)
      LeftBound28116.bound (LeftBound28116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47129⟩⟩) (rawTerms := some (Proof.Events109.exact28121RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28116.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31622.bound, LeftBound28116.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31622.bound, LeftBound28116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31622.actual selector witness, LeftBound28116.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31627

namespace LeftBound31631
def owner : Owner := ⟨.program ⟨257⟩, ⟨69489⟩⟩
def transferEvent : Nat := 31631
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31629 .coefficient, .predecessor 1 31630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31629 .coefficient)
      LeftBound31626.bound (LeftBound31626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31630 .coefficient)
      LeftBound27902.bound (LeftBound27902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events109.exact27909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31626.bound, LeftBound27902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31626.bound, LeftBound27902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31626.actual selector witness, LeftBound27902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31631

namespace LeftBound31632
def owner : Owner := ⟨.program ⟨257⟩, ⟨69489⟩⟩
def transferEvent : Nat := 31632
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31628 .summary, .result 27909 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31628 .summary)
      LeftBound31627.bound (LeftBound31627.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69488⟩⟩) (rawTerms := some (Proof.Events123.exact31628RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27909 .summary)
      LeftBound27904.bound (LeftBound27904.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49809⟩⟩) (rawTerms := some (Proof.Events109.exact27909RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27904.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31627.bound, LeftBound27904.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31627.bound, LeftBound27904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31627.actual selector witness, LeftBound27904.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31632

namespace LeftBound31636
def owner : Owner := ⟨.program ⟨257⟩, ⟨70974⟩⟩
def transferEvent : Nat := 31636
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 31634 .coefficient, .predecessor 1 31635 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31634 .coefficient)
      LeftBound31631.bound (LeftBound31631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events123.exact31633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound31631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound31631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31635 .coefficient)
      LeftBound27690.bound (LeftBound27690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events108.exact27697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31631.bound, LeftBound27690.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31631.bound, LeftBound27690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31631.actual selector witness, LeftBound27690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31636

namespace LeftBound31637
def owner : Owner := ⟨.program ⟨257⟩, ⟨70974⟩⟩
def transferEvent : Nat := 31637
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 31633 .summary, .result 27697 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 31633 .summary)
      LeftBound31632.bound (LeftBound31632.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69489⟩⟩) (rawTerms := some (Proof.Events123.exact31633RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound31632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 27697 .summary)
      LeftBound27692.bound (LeftBound27692.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70972⟩⟩) (rawTerms := some (Proof.Events108.exact27697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound27692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound31632.bound, LeftBound27692.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound31632.bound, LeftBound27692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound31632.actual selector witness, LeftBound27692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound31637

namespace LeftBound31643
def owner : Owner := ⟨.program ⟨257⟩, ⟨7401⟩⟩
def transferEvent : Nat := 31643
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 31641 .coefficient) (.predecessor 1 31642 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 31641 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 31642 .coefficient)
      LeftAuthority15986.bound (LeftAuthority15986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events062.exact15987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority15986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority15986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority15986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound31643

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
