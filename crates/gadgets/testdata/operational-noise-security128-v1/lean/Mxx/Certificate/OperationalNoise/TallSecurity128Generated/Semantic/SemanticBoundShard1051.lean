import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1018
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1021
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1025
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1029
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1032
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1036
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1043
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1050

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound157703
def owner : Owner := ⟨.program ⟨257⟩, ⟨23783⟩⟩
def transferEvent : Nat := 157703
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157701 .coefficient, .predecessor 1 157702 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157701 .coefficient)
      LeftBound157698.bound (LeftBound157698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157702 .coefficient)
      LeftBound156727.bound (LeftBound156727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156727.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157698.bound, LeftBound156727.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157698.bound, LeftBound156727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157698.actual selector witness, LeftBound156727.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157703

namespace LeftBound157704
def owner : Owner := ⟨.program ⟨257⟩, ⟨23783⟩⟩
def transferEvent : Nat := 157704
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157700 .summary, .result 156731 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157700 .summary)
      LeftBound157699.bound (LeftBound157699.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20563⟩⟩) (rawTerms := some (Proof.Events616.exact157700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 156731 .summary)
      LeftBound156730.bound (LeftBound156730.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23782⟩⟩) (rawTerms := some (Proof.Events612.exact156731RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound156730.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157699.bound, LeftBound156730.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157699.bound, LeftBound156730.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157699.actual selector witness, LeftBound156730.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157704

namespace LeftBound157708
def owner : Owner := ⟨.program ⟨257⟩, ⟨33803⟩⟩
def transferEvent : Nat := 157708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157706 .coefficient, .predecessor 1 157707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157706 .coefficient)
      LeftBound157703.bound (LeftBound157703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157703.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157707 .coefficient)
      LeftBound156245.bound (LeftBound156245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events610.exact156249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156245.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157703.bound, LeftBound156245.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157703.bound, LeftBound156245.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157703.actual selector witness, LeftBound156245.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157708

namespace LeftBound157709
def owner : Owner := ⟨.program ⟨257⟩, ⟨33803⟩⟩
def transferEvent : Nat := 157709
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157705 .summary, .result 156249 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157705 .summary)
      LeftBound157704.bound (LeftBound157704.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23783⟩⟩) (rawTerms := some (Proof.Events616.exact157705RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 156249 .summary)
      LeftBound156248.bound (LeftBound156248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33802⟩⟩) (rawTerms := some (Proof.Events610.exact156249RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound156248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157704.bound, LeftBound156248.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157704.bound, LeftBound156248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157704.actual selector witness, LeftBound156248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157709

namespace LeftBound157713
def owner : Owner := ⟨.program ⟨257⟩, ⟨52863⟩⟩
def transferEvent : Nat := 157713
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157711 .coefficient, .predecessor 1 157712 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157711 .coefficient)
      LeftBound157708.bound (LeftBound157708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157712 .coefficient)
      LeftBound155763.bound (LeftBound155763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events608.exact155767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155763.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157708.bound, LeftBound155763.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157708.bound, LeftBound155763.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157708.actual selector witness, LeftBound155763.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157713

namespace LeftBound157714
def owner : Owner := ⟨.program ⟨257⟩, ⟨52863⟩⟩
def transferEvent : Nat := 157714
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157710 .summary, .result 155767 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157710 .summary)
      LeftBound157709.bound (LeftBound157709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33803⟩⟩) (rawTerms := some (Proof.Events616.exact157710RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155767 .summary)
      LeftBound155766.bound (LeftBound155766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52862⟩⟩) (rawTerms := some (Proof.Events608.exact155767RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157709.bound, LeftBound155766.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157709.bound, LeftBound155766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157709.actual selector witness, LeftBound155766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157714

namespace LeftBound157718
def owner : Owner := ⟨.program ⟨257⟩, ⟨55843⟩⟩
def transferEvent : Nat := 157718
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157716 .coefficient, .predecessor 1 157717 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157716 .coefficient)
      LeftBound157713.bound (LeftBound157713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157717 .coefficient)
      LeftBound155281.bound (LeftBound155281.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events606.exact155285RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound155281.bound, RecordedBoundRefines] <;> decide)
      (LeftBound155281.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157713.bound, LeftBound155281.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157713.bound, LeftBound155281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157713.actual selector witness, LeftBound155281.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157718

namespace LeftBound157719
def owner : Owner := ⟨.program ⟨257⟩, ⟨55843⟩⟩
def transferEvent : Nat := 157719
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157715 .summary, .result 155285 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157715 .summary)
      LeftBound157714.bound (LeftBound157714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52863⟩⟩) (rawTerms := some (Proof.Events616.exact157715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 155285 .summary)
      LeftBound155284.bound (LeftBound155284.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55842⟩⟩) (rawTerms := some (Proof.Events606.exact155285RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound155284.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157714.bound, LeftBound155284.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157714.bound, LeftBound155284.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157714.actual selector witness, LeftBound155284.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157719

namespace LeftBound157723
def owner : Owner := ⟨.program ⟨257⟩, ⟨58823⟩⟩
def transferEvent : Nat := 157723
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157721 .coefficient, .predecessor 1 157722 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157721 .coefficient)
      LeftBound157718.bound (LeftBound157718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157718.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157722 .coefficient)
      LeftBound154799.bound (LeftBound154799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events604.exact154803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157718.bound, LeftBound154799.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157718.bound, LeftBound154799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157718.actual selector witness, LeftBound154799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157723

namespace LeftBound157724
def owner : Owner := ⟨.program ⟨257⟩, ⟨58823⟩⟩
def transferEvent : Nat := 157724
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157720 .summary, .result 154803 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157720 .summary)
      LeftBound157719.bound (LeftBound157719.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55843⟩⟩) (rawTerms := some (Proof.Events616.exact157720RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 154803 .summary)
      LeftBound154802.bound (LeftBound154802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58822⟩⟩) (rawTerms := some (Proof.Events604.exact154803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound154802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157719.bound, LeftBound154802.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157719.bound, LeftBound154802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157719.actual selector witness, LeftBound154802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157724

namespace LeftBound157728
def owner : Owner := ⟨.program ⟨257⟩, ⟨61803⟩⟩
def transferEvent : Nat := 157728
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157726 .coefficient, .predecessor 1 157727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157726 .coefficient)
      LeftBound157723.bound (LeftBound157723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157727 .coefficient)
      LeftBound154317.bound (LeftBound154317.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events602.exact154321RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound154317.bound, RecordedBoundRefines] <;> decide)
      (LeftBound154317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157723.bound, LeftBound154317.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157723.bound, LeftBound154317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157723.actual selector witness, LeftBound154317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157728

namespace LeftBound157729
def owner : Owner := ⟨.program ⟨257⟩, ⟨61803⟩⟩
def transferEvent : Nat := 157729
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157725 .summary, .result 154321 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157725 .summary)
      LeftBound157724.bound (LeftBound157724.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58823⟩⟩) (rawTerms := some (Proof.Events616.exact157725RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 154321 .summary)
      LeftBound154320.bound (LeftBound154320.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61802⟩⟩) (rawTerms := some (Proof.Events602.exact154321RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound154320.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157724.bound, LeftBound154320.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157724.bound, LeftBound154320.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157724.actual selector witness, LeftBound154320.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157729

namespace LeftBound157733
def owner : Owner := ⟨.program ⟨257⟩, ⟨64783⟩⟩
def transferEvent : Nat := 157733
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157731 .coefficient, .predecessor 1 157732 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157731 .coefficient)
      LeftBound157728.bound (LeftBound157728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157732 .coefficient)
      LeftBound153835.bound (LeftBound153835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events600.exact153839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157728.bound, LeftBound153835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157728.bound, LeftBound153835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157728.actual selector witness, LeftBound153835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157733

namespace LeftBound157734
def owner : Owner := ⟨.program ⟨257⟩, ⟨64783⟩⟩
def transferEvent : Nat := 157734
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157730 .summary, .result 153839 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157730 .summary)
      LeftBound157729.bound (LeftBound157729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61803⟩⟩) (rawTerms := some (Proof.Events616.exact157730RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 153839 .summary)
      LeftBound153838.bound (LeftBound153838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64782⟩⟩) (rawTerms := some (Proof.Events600.exact153839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound153838.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157729.bound, LeftBound153838.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157729.bound, LeftBound153838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157729.actual selector witness, LeftBound153838.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157734

namespace LeftBound157738
def owner : Owner := ⟨.program ⟨257⟩, ⟨69944⟩⟩
def transferEvent : Nat := 157738
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 157736 .coefficient, .predecessor 1 157737 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 157736 .coefficient)
      LeftBound157733.bound (LeftBound157733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events616.exact157735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound157733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound157733.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 157737 .coefficient)
      LeftBound153353.bound (LeftBound153353.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events599.exact153357RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound153353.bound, RecordedBoundRefines] <;> decide)
      (LeftBound153353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157733.bound, LeftBound153353.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157733.bound, LeftBound153353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157733.actual selector witness, LeftBound153353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157738

namespace LeftBound157739
def owner : Owner := ⟨.program ⟨257⟩, ⟨69944⟩⟩
def transferEvent : Nat := 157739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 157735 .summary, .result 153357 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 157735 .summary)
      LeftBound157734.bound (LeftBound157734.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64783⟩⟩) (rawTerms := some (Proof.Events616.exact157735RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound157734.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 153357 .summary)
      LeftBound153356.bound (LeftBound153356.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69943⟩⟩) (rawTerms := some (Proof.Events599.exact153357RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound153356.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound157734.bound, LeftBound153356.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound157734.bound, LeftBound153356.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound157734.actual selector witness, LeftBound153356.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound157739

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
