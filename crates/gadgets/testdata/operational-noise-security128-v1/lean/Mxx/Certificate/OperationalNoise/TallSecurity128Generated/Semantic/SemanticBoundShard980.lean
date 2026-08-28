import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard958
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard960
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard961
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard962
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard964
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard965
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard966
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard967
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard968
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard969
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard979

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound148582
def owner : Owner := ⟨.program ⟨257⟩, ⟨61673⟩⟩
def transferEvent : Nat := 148582
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148578 .summary, .result 147032 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148578 .summary)
      LeftBound148577.bound (LeftBound148577.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58693⟩⟩) (rawTerms := some (Proof.Events580.exact148578RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 147032 .summary)
      LeftBound147027.bound (LeftBound147027.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61672⟩⟩) (rawTerms := some (Proof.Events574.exact147032RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound147027.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148577.bound, LeftBound147027.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148577.bound, LeftBound147027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148577.actual selector witness, LeftBound147027.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148582

namespace LeftBound148586
def owner : Owner := ⟨.program ⟨257⟩, ⟨64653⟩⟩
def transferEvent : Nat := 148586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148584 .coefficient, .predecessor 1 148585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148584 .coefficient)
      LeftBound148581.bound (LeftBound148581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148585 .coefficient)
      LeftBound146813.bound (LeftBound146813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events573.exact146820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound146813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound146813.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148581.bound, LeftBound146813.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148581.bound, LeftBound146813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148581.actual selector witness, LeftBound146813.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148586

namespace LeftBound148587
def owner : Owner := ⟨.program ⟨257⟩, ⟨64653⟩⟩
def transferEvent : Nat := 148587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148583 .summary, .result 146820 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148583 .summary)
      LeftBound148582.bound (LeftBound148582.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61673⟩⟩) (rawTerms := some (Proof.Events580.exact148583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 146820 .summary)
      LeftBound146815.bound (LeftBound146815.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64652⟩⟩) (rawTerms := some (Proof.Events573.exact146820RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound146815.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148582.bound, LeftBound146815.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148582.bound, LeftBound146815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148582.actual selector witness, LeftBound146815.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148587

namespace LeftBound148591
def owner : Owner := ⟨.program ⟨257⟩, ⟨69614⟩⟩
def transferEvent : Nat := 148591
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148589 .coefficient, .predecessor 1 148590 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148589 .coefficient)
      LeftBound148586.bound (LeftBound148586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148590 .coefficient)
      LeftBound146601.bound (LeftBound146601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events572.exact146608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound146601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound146601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148586.bound, LeftBound146601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148586.bound, LeftBound146601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148586.actual selector witness, LeftBound146601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148591

namespace LeftBound148592
def owner : Owner := ⟨.program ⟨257⟩, ⟨69614⟩⟩
def transferEvent : Nat := 148592
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148588 .summary, .result 146608 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148588 .summary)
      LeftBound148587.bound (LeftBound148587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64653⟩⟩) (rawTerms := some (Proof.Events580.exact148588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 146608 .summary)
      LeftBound146603.bound (LeftBound146603.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69613⟩⟩) (rawTerms := some (Proof.Events572.exact146608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound146603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148587.bound, LeftBound146603.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148587.bound, LeftBound146603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148587.actual selector witness, LeftBound146603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148592

namespace LeftBound148596
def owner : Owner := ⟨.program ⟨257⟩, ⟨69615⟩⟩
def transferEvent : Nat := 148596
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148594 .coefficient, .predecessor 1 148595 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148594 .coefficient)
      LeftBound148591.bound (LeftBound148591.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148591.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148591.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148595 .coefficient)
      LeftBound146389.bound (LeftBound146389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events571.exact146396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound146389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound146389.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148591.bound, LeftBound146389.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148591.bound, LeftBound146389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148591.actual selector witness, LeftBound146389.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148596

namespace LeftBound148597
def owner : Owner := ⟨.program ⟨257⟩, ⟨69615⟩⟩
def transferEvent : Nat := 148597
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148593 .summary, .result 146396 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148593 .summary)
      LeftBound148592.bound (LeftBound148592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69614⟩⟩) (rawTerms := some (Proof.Events580.exact148593RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 146396 .summary)
      LeftBound146391.bound (LeftBound146391.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28112⟩⟩) (rawTerms := some (Proof.Events571.exact146396RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound146391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148592.bound, LeftBound146391.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148592.bound, LeftBound146391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148592.actual selector witness, LeftBound146391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148597

namespace LeftBound148601
def owner : Owner := ⟨.program ⟨257⟩, ⟨69616⟩⟩
def transferEvent : Nat := 148601
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148599 .coefficient, .predecessor 1 148600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148599 .coefficient)
      LeftBound148596.bound (LeftBound148596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148600 .coefficient)
      LeftBound146177.bound (LeftBound146177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events571.exact146184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound146177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound146177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148596.bound, LeftBound146177.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148596.bound, LeftBound146177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148596.actual selector witness, LeftBound146177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148601

namespace LeftBound148602
def owner : Owner := ⟨.program ⟨257⟩, ⟨69616⟩⟩
def transferEvent : Nat := 148602
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148598 .summary, .result 146184 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148598 .summary)
      LeftBound148597.bound (LeftBound148597.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69615⟩⟩) (rawTerms := some (Proof.Events580.exact148598RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 146184 .summary)
      LeftBound146179.bound (LeftBound146179.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30792⟩⟩) (rawTerms := some (Proof.Events571.exact146184RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound146179.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148597.bound, LeftBound146179.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148597.bound, LeftBound146179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148597.actual selector witness, LeftBound146179.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148602

namespace LeftBound148606
def owner : Owner := ⟨.program ⟨257⟩, ⟨69617⟩⟩
def transferEvent : Nat := 148606
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148604 .coefficient, .predecessor 1 148605 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148604 .coefficient)
      LeftBound148601.bound (LeftBound148601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148605 .coefficient)
      LeftBound145965.bound (LeftBound145965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events570.exact145972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145965.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148601.bound, LeftBound145965.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148601.bound, LeftBound145965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148601.actual selector witness, LeftBound145965.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148606

namespace LeftBound148607
def owner : Owner := ⟨.program ⟨257⟩, ⟨69617⟩⟩
def transferEvent : Nat := 148607
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148603 .summary, .result 145972 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148603 .summary)
      LeftBound148602.bound (LeftBound148602.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69616⟩⟩) (rawTerms := some (Proof.Events580.exact148603RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145972 .summary)
      LeftBound145967.bound (LeftBound145967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36452⟩⟩) (rawTerms := some (Proof.Events570.exact145972RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148602.bound, LeftBound145967.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148602.bound, LeftBound145967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148602.actual selector witness, LeftBound145967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148607

namespace LeftBound148611
def owner : Owner := ⟨.program ⟨257⟩, ⟨69618⟩⟩
def transferEvent : Nat := 148611
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148609 .coefficient, .predecessor 1 148610 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148609 .coefficient)
      LeftBound148606.bound (LeftBound148606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148610 .coefficient)
      LeftBound145753.bound (LeftBound145753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events569.exact145760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148606.bound, LeftBound145753.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148606.bound, LeftBound145753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148606.actual selector witness, LeftBound145753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148611

namespace LeftBound148612
def owner : Owner := ⟨.program ⟨257⟩, ⟨69618⟩⟩
def transferEvent : Nat := 148612
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148608 .summary, .result 145760 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148608 .summary)
      LeftBound148607.bound (LeftBound148607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69617⟩⟩) (rawTerms := some (Proof.Events580.exact148608RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145760 .summary)
      LeftBound145755.bound (LeftBound145755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39132⟩⟩) (rawTerms := some (Proof.Events569.exact145760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148607.bound, LeftBound145755.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148607.bound, LeftBound145755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148607.actual selector witness, LeftBound145755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148612

namespace LeftBound148616
def owner : Owner := ⟨.program ⟨257⟩, ⟨69619⟩⟩
def transferEvent : Nat := 148616
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148614 .coefficient, .predecessor 1 148615 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148614 .coefficient)
      LeftBound148611.bound (LeftBound148611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148615 .coefficient)
      LeftBound145541.bound (LeftBound145541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events568.exact145548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148611.bound, LeftBound145541.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148611.bound, LeftBound145541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148611.actual selector witness, LeftBound145541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148616

namespace LeftBound148617
def owner : Owner := ⟨.program ⟨257⟩, ⟨69619⟩⟩
def transferEvent : Nat := 148617
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 148613 .summary, .result 145548 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 148613 .summary)
      LeftBound148612.bound (LeftBound148612.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69618⟩⟩) (rawTerms := some (Proof.Events580.exact148613RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound148612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 145548 .summary)
      LeftBound145543.bound (LeftBound145543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41812⟩⟩) (rawTerms := some (Proof.Events568.exact145548RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound145543.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148612.bound, LeftBound145543.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148612.bound, LeftBound145543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148612.actual selector witness, LeftBound145543.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148617

namespace LeftBound148621
def owner : Owner := ⟨.program ⟨257⟩, ⟨69620⟩⟩
def transferEvent : Nat := 148621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 148619 .coefficient, .predecessor 1 148620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 148619 .coefficient)
      LeftBound148616.bound (LeftBound148616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events580.exact148618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 148620 .coefficient)
      LeftBound145329.bound (LeftBound145329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events567.exact145336RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound145329.bound, RecordedBoundRefines] <;> decide)
      (LeftBound145329.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound148616.bound, LeftBound145329.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148616.bound, LeftBound145329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound148616.actual selector witness, LeftBound145329.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound148621

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
