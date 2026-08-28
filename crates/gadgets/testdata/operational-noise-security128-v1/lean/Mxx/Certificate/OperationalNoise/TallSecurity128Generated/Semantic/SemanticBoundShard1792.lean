import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1766
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1767
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1768
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1769
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1770
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1771
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1772
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1773
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1774
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1775
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1791

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound265602
def owner : Owner := ⟨.program ⟨257⟩, ⟨69774⟩⟩
def transferEvent : Nat := 265602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265598 .summary, .result 263184 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265598 .summary)
      LeftBound265597.bound (LeftBound265597.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69773⟩⟩) (rawTerms := some (Proof.Events1037.exact265598RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 263184 .summary)
      LeftBound263179.bound (LeftBound263179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30842⟩⟩) (rawTerms := some (Proof.Events1028.exact263184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound263179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265597.bound, LeftBound263179.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265597.bound, LeftBound263179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265597.actual selector witness, LeftBound263179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265602

namespace LeftBound265606
def owner : Owner := ⟨.program ⟨257⟩, ⟨69775⟩⟩
def transferEvent : Nat := 265606
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265604 .coefficient, .predecessor 1 265605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265604 .coefficient)
      LeftBound265601.bound (LeftBound265601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265605 .coefficient)
      LeftBound262965.bound (LeftBound262965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1027.exact262972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound262965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound262965.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265601.bound, LeftBound262965.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265601.bound, LeftBound262965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265601.actual selector witness, LeftBound262965.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265606

namespace LeftBound265607
def owner : Owner := ⟨.program ⟨257⟩, ⟨69775⟩⟩
def transferEvent : Nat := 265607
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265603 .summary, .result 262972 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265603 .summary)
      LeftBound265602.bound (LeftBound265602.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69774⟩⟩) (rawTerms := some (Proof.Events1037.exact265603RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 262972 .summary)
      LeftBound262967.bound (LeftBound262967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36502⟩⟩) (rawTerms := some (Proof.Events1027.exact262972RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound262967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265602.bound, LeftBound262967.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265602.bound, LeftBound262967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265602.actual selector witness, LeftBound262967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265607

namespace LeftBound265611
def owner : Owner := ⟨.program ⟨257⟩, ⟨69776⟩⟩
def transferEvent : Nat := 265611
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265609 .coefficient, .predecessor 1 265610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265609 .coefficient)
      LeftBound265606.bound (LeftBound265606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265610 .coefficient)
      LeftBound262753.bound (LeftBound262753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1026.exact262760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound262753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound262753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265606.bound, LeftBound262753.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265606.bound, LeftBound262753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265606.actual selector witness, LeftBound262753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265611

namespace LeftBound265612
def owner : Owner := ⟨.program ⟨257⟩, ⟨69776⟩⟩
def transferEvent : Nat := 265612
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265608 .summary, .result 262760 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265608 .summary)
      LeftBound265607.bound (LeftBound265607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69775⟩⟩) (rawTerms := some (Proof.Events1037.exact265608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 262760 .summary)
      LeftBound262755.bound (LeftBound262755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39182⟩⟩) (rawTerms := some (Proof.Events1026.exact262760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound262755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265607.bound, LeftBound262755.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265607.bound, LeftBound262755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265607.actual selector witness, LeftBound262755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265612

namespace LeftBound265616
def owner : Owner := ⟨.program ⟨257⟩, ⟨69777⟩⟩
def transferEvent : Nat := 265616
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265614 .coefficient, .predecessor 1 265615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265614 .coefficient)
      LeftBound265611.bound (LeftBound265611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265615 .coefficient)
      LeftBound262541.bound (LeftBound262541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1025.exact262548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound262541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound262541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265611.bound, LeftBound262541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265611.bound, LeftBound262541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265611.actual selector witness, LeftBound262541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265616

namespace LeftBound265617
def owner : Owner := ⟨.program ⟨257⟩, ⟨69777⟩⟩
def transferEvent : Nat := 265617
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265613 .summary, .result 262548 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265613 .summary)
      LeftBound265612.bound (LeftBound265612.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69776⟩⟩) (rawTerms := some (Proof.Events1037.exact265613RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 262548 .summary)
      LeftBound262543.bound (LeftBound262543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41862⟩⟩) (rawTerms := some (Proof.Events1025.exact262548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound262543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265612.bound, LeftBound262543.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265612.bound, LeftBound262543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265612.actual selector witness, LeftBound262543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265617

namespace LeftBound265621
def owner : Owner := ⟨.program ⟨257⟩, ⟨69778⟩⟩
def transferEvent : Nat := 265621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265619 .coefficient, .predecessor 1 265620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265619 .coefficient)
      LeftBound265616.bound (LeftBound265616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265620 .coefficient)
      LeftBound262329.bound (LeftBound262329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1024.exact262336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound262329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound262329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265616.bound, LeftBound262329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265616.bound, LeftBound262329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265616.actual selector witness, LeftBound262329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265621

namespace LeftBound265622
def owner : Owner := ⟨.program ⟨257⟩, ⟨69778⟩⟩
def transferEvent : Nat := 265622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265618 .summary, .result 262336 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265618 .summary)
      LeftBound265617.bound (LeftBound265617.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69777⟩⟩) (rawTerms := some (Proof.Events1037.exact265618RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 262336 .summary)
      LeftBound262331.bound (LeftBound262331.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44542⟩⟩) (rawTerms := some (Proof.Events1024.exact262336RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound262331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265617.bound, LeftBound262331.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265617.bound, LeftBound262331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265617.actual selector witness, LeftBound262331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265622

namespace LeftBound265626
def owner : Owner := ⟨.program ⟨257⟩, ⟨69779⟩⟩
def transferEvent : Nat := 265626
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265624 .coefficient, .predecessor 1 265625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265624 .coefficient)
      LeftBound265621.bound (LeftBound265621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265621.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265625 .coefficient)
      LeftBound262117.bound (LeftBound262117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1023.exact262124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound262117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound262117.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265621.bound, LeftBound262117.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265621.bound, LeftBound262117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265621.actual selector witness, LeftBound262117.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265626

namespace LeftBound265627
def owner : Owner := ⟨.program ⟨257⟩, ⟨69779⟩⟩
def transferEvent : Nat := 265627
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265623 .summary, .result 262124 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265623 .summary)
      LeftBound265622.bound (LeftBound265622.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69778⟩⟩) (rawTerms := some (Proof.Events1037.exact265623RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 262124 .summary)
      LeftBound262119.bound (LeftBound262119.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47222⟩⟩) (rawTerms := some (Proof.Events1023.exact262124RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound262119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265622.bound, LeftBound262119.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265622.bound, LeftBound262119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265622.actual selector witness, LeftBound262119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265627

namespace LeftBound265631
def owner : Owner := ⟨.program ⟨257⟩, ⟨69780⟩⟩
def transferEvent : Nat := 265631
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265629 .coefficient, .predecessor 1 265630 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265629 .coefficient)
      LeftBound265626.bound (LeftBound265626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265630 .coefficient)
      LeftBound261905.bound (LeftBound261905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1023.exact261912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265626.bound, LeftBound261905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265626.bound, LeftBound261905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265626.actual selector witness, LeftBound261905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265631

namespace LeftBound265632
def owner : Owner := ⟨.program ⟨257⟩, ⟨69780⟩⟩
def transferEvent : Nat := 265632
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265628 .summary, .result 261912 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265628 .summary)
      LeftBound265627.bound (LeftBound265627.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69779⟩⟩) (rawTerms := some (Proof.Events1037.exact265628RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261912 .summary)
      LeftBound261907.bound (LeftBound261907.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49902⟩⟩) (rawTerms := some (Proof.Events1023.exact261912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound261907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265627.bound, LeftBound261907.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265627.bound, LeftBound261907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265627.actual selector witness, LeftBound261907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265632

namespace LeftBound265636
def owner : Owner := ⟨.program ⟨257⟩, ⟨71088⟩⟩
def transferEvent : Nat := 265636
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265634 .coefficient, .predecessor 1 265635 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265634 .coefficient)
      LeftBound265631.bound (LeftBound265631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1037.exact265633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265631.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265635 .coefficient)
      LeftBound261693.bound (LeftBound261693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265631.bound, LeftBound261693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265631.bound, LeftBound261693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265631.actual selector witness, LeftBound261693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265636

namespace LeftBound265637
def owner : Owner := ⟨.program ⟨257⟩, ⟨71088⟩⟩
def transferEvent : Nat := 265637
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265633 .summary, .result 261700 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265633 .summary)
      LeftBound265632.bound (LeftBound265632.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69780⟩⟩) (rawTerms := some (Proof.Events1037.exact265633RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261700 .summary)
      LeftBound261695.bound (LeftBound261695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71086⟩⟩) (rawTerms := some (Proof.Events1022.exact261700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound261695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265632.bound, LeftBound261695.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265632.bound, LeftBound261695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265632.actual selector witness, LeftBound261695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265637

namespace LeftBound265643
def owner : Owner := ⟨.program ⟨257⟩, ⟨7417⟩⟩
def transferEvent : Nat := 265643
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265641 .coefficient) (.predecessor 1 265642 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265641 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265642 .coefficient)
      LeftAuthority16626.bound (LeftAuthority16626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events064.exact16627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16626.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16626.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16626.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265643

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
