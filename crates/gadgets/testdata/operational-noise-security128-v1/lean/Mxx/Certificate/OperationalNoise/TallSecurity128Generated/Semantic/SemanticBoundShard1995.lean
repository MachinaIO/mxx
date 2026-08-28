import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1898
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1969
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1970
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1971
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1973
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1974
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1975
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1977
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1994

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound294821
def owner : Owner := ⟨.program ⟨257⟩, ⟨69696⟩⟩
def transferEvent : Nat := 294821
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294817 .summary, .result 292186 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294817 .summary)
      LeftBound294816.bound (LeftBound294816.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69695⟩⟩) (rawTerms := some (Proof.Events1151.exact294817RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 292186 .summary)
      LeftBound292181.bound (LeftBound292181.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36477⟩⟩) (rawTerms := some (Proof.Events1141.exact292186RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound292181.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294816.bound, LeftBound292181.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294816.bound, LeftBound292181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294816.actual selector witness, LeftBound292181.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294821

namespace LeftBound294825
def owner : Owner := ⟨.program ⟨257⟩, ⟨69697⟩⟩
def transferEvent : Nat := 294825
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294823 .coefficient, .predecessor 1 294824 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294823 .coefficient)
      LeftBound294820.bound (LeftBound294820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294824 .coefficient)
      LeftBound291967.bound (LeftBound291967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1140.exact291974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound291967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound291967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294820.bound, LeftBound291967.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294820.bound, LeftBound291967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294820.actual selector witness, LeftBound291967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294825

namespace LeftBound294826
def owner : Owner := ⟨.program ⟨257⟩, ⟨69697⟩⟩
def transferEvent : Nat := 294826
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294822 .summary, .result 291974 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294822 .summary)
      LeftBound294821.bound (LeftBound294821.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69696⟩⟩) (rawTerms := some (Proof.Events1151.exact294822RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 291974 .summary)
      LeftBound291969.bound (LeftBound291969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39157⟩⟩) (rawTerms := some (Proof.Events1140.exact291974RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound291969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294821.bound, LeftBound291969.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294821.bound, LeftBound291969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294821.actual selector witness, LeftBound291969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294826

namespace LeftBound294830
def owner : Owner := ⟨.program ⟨257⟩, ⟨69698⟩⟩
def transferEvent : Nat := 294830
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294828 .coefficient, .predecessor 1 294829 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294828 .coefficient)
      LeftBound294825.bound (LeftBound294825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294827RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294829 .coefficient)
      LeftBound291755.bound (LeftBound291755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1139.exact291762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound291755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound291755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294825.bound, LeftBound291755.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294825.bound, LeftBound291755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294825.actual selector witness, LeftBound291755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294830

namespace LeftBound294831
def owner : Owner := ⟨.program ⟨257⟩, ⟨69698⟩⟩
def transferEvent : Nat := 294831
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294827 .summary, .result 291762 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294827 .summary)
      LeftBound294826.bound (LeftBound294826.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69697⟩⟩) (rawTerms := some (Proof.Events1151.exact294827RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294826.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 291762 .summary)
      LeftBound291757.bound (LeftBound291757.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41837⟩⟩) (rawTerms := some (Proof.Events1139.exact291762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound291757.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294826.bound, LeftBound291757.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294826.bound, LeftBound291757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294826.actual selector witness, LeftBound291757.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294831

namespace LeftBound294835
def owner : Owner := ⟨.program ⟨257⟩, ⟨69699⟩⟩
def transferEvent : Nat := 294835
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294833 .coefficient, .predecessor 1 294834 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294833 .coefficient)
      LeftBound294830.bound (LeftBound294830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294830.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294834 .coefficient)
      LeftBound291543.bound (LeftBound291543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1138.exact291550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound291543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound291543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294830.bound, LeftBound291543.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294830.bound, LeftBound291543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294830.actual selector witness, LeftBound291543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294835

namespace LeftBound294836
def owner : Owner := ⟨.program ⟨257⟩, ⟨69699⟩⟩
def transferEvent : Nat := 294836
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294832 .summary, .result 291550 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294832 .summary)
      LeftBound294831.bound (LeftBound294831.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69698⟩⟩) (rawTerms := some (Proof.Events1151.exact294832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 291550 .summary)
      LeftBound291545.bound (LeftBound291545.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44517⟩⟩) (rawTerms := some (Proof.Events1138.exact291550RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound291545.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294831.bound, LeftBound291545.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294831.bound, LeftBound291545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294831.actual selector witness, LeftBound291545.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294836

namespace LeftBound294840
def owner : Owner := ⟨.program ⟨257⟩, ⟨69700⟩⟩
def transferEvent : Nat := 294840
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294838 .coefficient, .predecessor 1 294839 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294838 .coefficient)
      LeftBound294835.bound (LeftBound294835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294839 .coefficient)
      LeftBound291331.bound (LeftBound291331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1138.exact291338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound291331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound291331.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294835.bound, LeftBound291331.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294835.bound, LeftBound291331.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294835.actual selector witness, LeftBound291331.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294840

namespace LeftBound294841
def owner : Owner := ⟨.program ⟨257⟩, ⟨69700⟩⟩
def transferEvent : Nat := 294841
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294837 .summary, .result 291338 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294837 .summary)
      LeftBound294836.bound (LeftBound294836.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69699⟩⟩) (rawTerms := some (Proof.Events1151.exact294837RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 291338 .summary)
      LeftBound291333.bound (LeftBound291333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47197⟩⟩) (rawTerms := some (Proof.Events1138.exact291338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound291333.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294836.bound, LeftBound291333.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294836.bound, LeftBound291333.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294836.actual selector witness, LeftBound291333.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294841

namespace LeftBound294845
def owner : Owner := ⟨.program ⟨257⟩, ⟨69701⟩⟩
def transferEvent : Nat := 294845
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294843 .coefficient, .predecessor 1 294844 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294843 .coefficient)
      LeftBound294840.bound (LeftBound294840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294844 .coefficient)
      LeftBound291119.bound (LeftBound291119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1137.exact291126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound291119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound291119.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294840.bound, LeftBound291119.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294840.bound, LeftBound291119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294840.actual selector witness, LeftBound291119.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294845

namespace LeftBound294846
def owner : Owner := ⟨.program ⟨257⟩, ⟨69701⟩⟩
def transferEvent : Nat := 294846
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294842 .summary, .result 291126 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294842 .summary)
      LeftBound294841.bound (LeftBound294841.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69700⟩⟩) (rawTerms := some (Proof.Events1151.exact294842RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 291126 .summary)
      LeftBound291121.bound (LeftBound291121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49877⟩⟩) (rawTerms := some (Proof.Events1137.exact291126RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound291121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294841.bound, LeftBound291121.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294841.bound, LeftBound291121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294841.actual selector witness, LeftBound291121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294846

namespace LeftBound294850
def owner : Owner := ⟨.program ⟨257⟩, ⟨71054⟩⟩
def transferEvent : Nat := 294850
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294848 .coefficient, .predecessor 1 294849 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294848 .coefficient)
      LeftBound294845.bound (LeftBound294845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294849 .coefficient)
      LeftBound290907.bound (LeftBound290907.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1136.exact290914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound290907.bound, RecordedBoundRefines] <;> decide)
      (LeftBound290907.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294845.bound, LeftBound290907.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294845.bound, LeftBound290907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294845.actual selector witness, LeftBound290907.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294850

namespace LeftBound294851
def owner : Owner := ⟨.program ⟨257⟩, ⟨71054⟩⟩
def transferEvent : Nat := 294851
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 294847 .summary, .result 290914 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 294847 .summary)
      LeftBound294846.bound (LeftBound294846.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69701⟩⟩) (rawTerms := some (Proof.Events1151.exact294847RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound294846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 290914 .summary)
      LeftBound290909.bound (LeftBound290909.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71052⟩⟩) (rawTerms := some (Proof.Events1136.exact290914RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound290909.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294846.bound, LeftBound290909.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294846.bound, LeftBound290909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294846.actual selector witness, LeftBound290909.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294851

namespace LeftBound294857
def owner : Owner := ⟨.program ⟨257⟩, ⟨7419⟩⟩
def transferEvent : Nat := 294857
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 294855 .coefficient) (.predecessor 1 294856 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294855 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294856 .coefficient)
      LeftAuthority16706.bound (LeftAuthority16706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftAuthority16706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority16706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftAuthority16706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound294857

namespace LeftBound294862
def owner : Owner := ⟨.program ⟨257⟩, ⟨9241⟩⟩
def transferEvent : Nat := 294862
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294860 .coefficient, .predecessor 1 294861 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294860 .coefficient)
      LeftBound294857.bound (LeftBound294857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294861 .coefficient)
      LeftBound280651.bound (LeftBound280651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1096.exact280653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280651.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294857.bound, LeftBound280651.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294857.bound, LeftBound280651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294857.actual selector witness, LeftBound280651.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294862

namespace LeftBound294866
def owner : Owner := ⟨.program ⟨257⟩, ⟨9242⟩⟩
def transferEvent : Nat := 294866
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 294864 .coefficient, .predecessor 1 294865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 294864 .coefficient)
      LeftBound294862.bound (LeftBound294862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound294862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound294862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 294865 .coefficient)
      LeftAuthority294853.bound (LeftAuthority294853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1151.exact294854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority294853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority294853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound294862.bound, LeftAuthority294853.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound294862.bound, LeftAuthority294853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound294862.actual selector witness, LeftAuthority294853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound294866

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
