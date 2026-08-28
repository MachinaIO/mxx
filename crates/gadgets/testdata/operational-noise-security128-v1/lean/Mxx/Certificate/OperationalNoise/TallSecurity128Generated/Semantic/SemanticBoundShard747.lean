import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard702
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard706
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard710
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard713
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard717
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard721
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard724
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard728
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard732
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard746

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound113839
def owner : Owner := ⟨.program ⟨257⟩, ⟨52987⟩⟩
def transferEvent : Nat := 113839
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113835 .summary, .result 111892 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113835 .summary)
      LeftBound113834.bound (LeftBound113834.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33927⟩⟩) (rawTerms := some (Proof.Events444.exact113835RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111892 .summary)
      LeftBound111891.bound (LeftBound111891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52986⟩⟩) (rawTerms := some (Proof.Events437.exact111892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113834.bound, LeftBound111891.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113834.bound, LeftBound111891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113834.actual selector witness, LeftBound111891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113839

namespace LeftBound113843
def owner : Owner := ⟨.program ⟨257⟩, ⟨55967⟩⟩
def transferEvent : Nat := 113843
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113841 .coefficient, .predecessor 1 113842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113841 .coefficient)
      LeftBound113838.bound (LeftBound113838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113842 .coefficient)
      LeftBound111406.bound (LeftBound111406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events435.exact111410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113838.bound, LeftBound111406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113838.bound, LeftBound111406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113838.actual selector witness, LeftBound111406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113843

namespace LeftBound113844
def owner : Owner := ⟨.program ⟨257⟩, ⟨55967⟩⟩
def transferEvent : Nat := 113844
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113840 .summary, .result 111410 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113840 .summary)
      LeftBound113839.bound (LeftBound113839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52987⟩⟩) (rawTerms := some (Proof.Events444.exact113840RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111410 .summary)
      LeftBound111409.bound (LeftBound111409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55966⟩⟩) (rawTerms := some (Proof.Events435.exact111410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111409.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113839.bound, LeftBound111409.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113839.bound, LeftBound111409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113839.actual selector witness, LeftBound111409.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113844

namespace LeftBound113848
def owner : Owner := ⟨.program ⟨257⟩, ⟨58947⟩⟩
def transferEvent : Nat := 113848
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113846 .coefficient, .predecessor 1 113847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113846 .coefficient)
      LeftBound113843.bound (LeftBound113843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113847 .coefficient)
      LeftBound110924.bound (LeftBound110924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events433.exact110928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110924.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113843.bound, LeftBound110924.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113843.bound, LeftBound110924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113843.actual selector witness, LeftBound110924.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113848

namespace LeftBound113849
def owner : Owner := ⟨.program ⟨257⟩, ⟨58947⟩⟩
def transferEvent : Nat := 113849
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113845 .summary, .result 110928 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113845 .summary)
      LeftBound113844.bound (LeftBound113844.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55967⟩⟩) (rawTerms := some (Proof.Events444.exact113845RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110928 .summary)
      LeftBound110927.bound (LeftBound110927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58946⟩⟩) (rawTerms := some (Proof.Events433.exact110928RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113844.bound, LeftBound110927.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113844.bound, LeftBound110927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113844.actual selector witness, LeftBound110927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113849

namespace LeftBound113853
def owner : Owner := ⟨.program ⟨257⟩, ⟨61927⟩⟩
def transferEvent : Nat := 113853
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113851 .coefficient, .predecessor 1 113852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113851 .coefficient)
      LeftBound113848.bound (LeftBound113848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113852 .coefficient)
      LeftBound110442.bound (LeftBound110442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events431.exact110446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound110442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound110442.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113848.bound, LeftBound110442.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113848.bound, LeftBound110442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113848.actual selector witness, LeftBound110442.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113853

namespace LeftBound113854
def owner : Owner := ⟨.program ⟨257⟩, ⟨61927⟩⟩
def transferEvent : Nat := 113854
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113850 .summary, .result 110446 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113850 .summary)
      LeftBound113849.bound (LeftBound113849.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58947⟩⟩) (rawTerms := some (Proof.Events444.exact113850RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 110446 .summary)
      LeftBound110445.bound (LeftBound110445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61926⟩⟩) (rawTerms := some (Proof.Events431.exact110446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound110445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113849.bound, LeftBound110445.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113849.bound, LeftBound110445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113849.actual selector witness, LeftBound110445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113854

namespace LeftBound113858
def owner : Owner := ⟨.program ⟨257⟩, ⟨64907⟩⟩
def transferEvent : Nat := 113858
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113856 .coefficient, .predecessor 1 113857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113856 .coefficient)
      LeftBound113853.bound (LeftBound113853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113857 .coefficient)
      LeftBound109960.bound (LeftBound109960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events429.exact109964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113853.bound, LeftBound109960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113853.bound, LeftBound109960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113853.actual selector witness, LeftBound109960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113858

namespace LeftBound113859
def owner : Owner := ⟨.program ⟨257⟩, ⟨64907⟩⟩
def transferEvent : Nat := 113859
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113855 .summary, .result 109964 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113855 .summary)
      LeftBound113854.bound (LeftBound113854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61927⟩⟩) (rawTerms := some (Proof.Events444.exact113855RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109964 .summary)
      LeftBound109963.bound (LeftBound109963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64906⟩⟩) (rawTerms := some (Proof.Events429.exact109964RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113854.bound, LeftBound109963.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113854.bound, LeftBound109963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113854.actual selector witness, LeftBound109963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113859

namespace LeftBound113863
def owner : Owner := ⟨.program ⟨257⟩, ⟨70260⟩⟩
def transferEvent : Nat := 113863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113861 .coefficient, .predecessor 1 113862 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113861 .coefficient)
      LeftBound113858.bound (LeftBound113858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113862 .coefficient)
      LeftBound109478.bound (LeftBound109478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events427.exact109482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound109478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound109478.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113858.bound, LeftBound109478.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113858.bound, LeftBound109478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113858.actual selector witness, LeftBound109478.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113863

namespace LeftBound113864
def owner : Owner := ⟨.program ⟨257⟩, ⟨70260⟩⟩
def transferEvent : Nat := 113864
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113860 .summary, .result 109482 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113860 .summary)
      LeftBound113859.bound (LeftBound113859.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64907⟩⟩) (rawTerms := some (Proof.Events444.exact113860RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109482 .summary)
      LeftBound109481.bound (LeftBound109481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70259⟩⟩) (rawTerms := some (Proof.Events427.exact109482RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound109481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113859.bound, LeftBound109481.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113859.bound, LeftBound109481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113859.actual selector witness, LeftBound109481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113864

namespace LeftBound113868
def owner : Owner := ⟨.program ⟨257⟩, ⟨70261⟩⟩
def transferEvent : Nat := 113868
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113866 .coefficient, .predecessor 1 113867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113866 .coefficient)
      LeftBound113863.bound (LeftBound113863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113867 .coefficient)
      LeftBound108996.bound (LeftBound108996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events425.exact109000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound108996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound108996.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113863.bound, LeftBound108996.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113863.bound, LeftBound108996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113863.actual selector witness, LeftBound108996.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113868

namespace LeftBound113869
def owner : Owner := ⟨.program ⟨257⟩, ⟨70261⟩⟩
def transferEvent : Nat := 113869
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113865 .summary, .result 109000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113865 .summary)
      LeftBound113864.bound (LeftBound113864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70260⟩⟩) (rawTerms := some (Proof.Events444.exact113865RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 109000 .summary)
      LeftBound108999.bound (LeftBound108999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28317⟩⟩) (rawTerms := some (Proof.Events425.exact109000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound108999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113864.bound, LeftBound108999.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113864.bound, LeftBound108999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113864.actual selector witness, LeftBound108999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113869

namespace LeftBound113873
def owner : Owner := ⟨.program ⟨257⟩, ⟨70262⟩⟩
def transferEvent : Nat := 113873
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113871 .coefficient, .predecessor 1 113872 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113871 .coefficient)
      LeftBound113868.bound (LeftBound113868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113872 .coefficient)
      LeftBound108514.bound (LeftBound108514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events423.exact108518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound108514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound108514.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113868.bound, LeftBound108514.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113868.bound, LeftBound108514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113868.actual selector witness, LeftBound108514.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113873

namespace LeftBound113874
def owner : Owner := ⟨.program ⟨257⟩, ⟨70262⟩⟩
def transferEvent : Nat := 113874
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 113870 .summary, .result 108518 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 113870 .summary)
      LeftBound113869.bound (LeftBound113869.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70261⟩⟩) (rawTerms := some (Proof.Events444.exact113870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound113869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 108518 .summary)
      LeftBound108517.bound (LeftBound108517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30997⟩⟩) (rawTerms := some (Proof.Events423.exact108518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound108517.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113869.bound, LeftBound108517.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113869.bound, LeftBound108517.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113869.actual selector witness, LeftBound108517.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113874

namespace LeftBound113878
def owner : Owner := ⟨.program ⟨257⟩, ⟨70263⟩⟩
def transferEvent : Nat := 113878
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 113876 .coefficient, .predecessor 1 113877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 113876 .coefficient)
      LeftBound113873.bound (LeftBound113873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events444.exact113875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound113873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound113873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 113877 .coefficient)
      LeftBound108032.bound (LeftBound108032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events422.exact108036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound108032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound108032.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound113873.bound, LeftBound108032.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound113873.bound, LeftBound108032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound113873.actual selector witness, LeftBound108032.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound113878

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
