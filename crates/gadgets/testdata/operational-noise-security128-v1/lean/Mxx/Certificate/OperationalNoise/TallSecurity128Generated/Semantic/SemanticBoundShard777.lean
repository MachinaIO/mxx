import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard757
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard758
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard759
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard761
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard762
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard763
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard765
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard766
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard767
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard776

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound119327
def owner : Owner := ⟨.program ⟨257⟩, ⟨58941⟩⟩
def transferEvent : Nat := 119327
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119323 .summary, .result 117994 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119323 .summary)
      LeftBound119322.bound (LeftBound119322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55961⟩⟩) (rawTerms := some (Proof.Events466.exact119323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 117994 .summary)
      LeftBound117989.bound (LeftBound117989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58940⟩⟩) (rawTerms := some (Proof.Events460.exact117994RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound117989.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119322.bound, LeftBound117989.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119322.bound, LeftBound117989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119322.actual selector witness, LeftBound117989.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119327

namespace LeftBound119331
def owner : Owner := ⟨.program ⟨257⟩, ⟨61921⟩⟩
def transferEvent : Nat := 119331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119329 .coefficient, .predecessor 1 119330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119329 .coefficient)
      LeftBound119326.bound (LeftBound119326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119330 .coefficient)
      LeftBound117775.bound (LeftBound117775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events460.exact117782RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound117775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound117775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119326.bound, LeftBound117775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119326.bound, LeftBound117775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119326.actual selector witness, LeftBound117775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119331

namespace LeftBound119332
def owner : Owner := ⟨.program ⟨257⟩, ⟨61921⟩⟩
def transferEvent : Nat := 119332
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119328 .summary, .result 117782 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119328 .summary)
      LeftBound119327.bound (LeftBound119327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58941⟩⟩) (rawTerms := some (Proof.Events466.exact119328RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119327.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 117782 .summary)
      LeftBound117777.bound (LeftBound117777.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61920⟩⟩) (rawTerms := some (Proof.Events460.exact117782RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound117777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119327.bound, LeftBound117777.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119327.bound, LeftBound117777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119327.actual selector witness, LeftBound117777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119332

namespace LeftBound119336
def owner : Owner := ⟨.program ⟨257⟩, ⟨64901⟩⟩
def transferEvent : Nat := 119336
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119334 .coefficient, .predecessor 1 119335 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119334 .coefficient)
      LeftBound119331.bound (LeftBound119331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119335 .coefficient)
      LeftBound117563.bound (LeftBound117563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events459.exact117570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound117563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound117563.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119331.bound, LeftBound117563.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119331.bound, LeftBound117563.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119331.actual selector witness, LeftBound117563.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119336

namespace LeftBound119337
def owner : Owner := ⟨.program ⟨257⟩, ⟨64901⟩⟩
def transferEvent : Nat := 119337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119333 .summary, .result 117570 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119333 .summary)
      LeftBound119332.bound (LeftBound119332.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61921⟩⟩) (rawTerms := some (Proof.Events466.exact119333RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 117570 .summary)
      LeftBound117565.bound (LeftBound117565.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64900⟩⟩) (rawTerms := some (Proof.Events459.exact117570RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound117565.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119332.bound, LeftBound117565.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119332.bound, LeftBound117565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119332.actual selector witness, LeftBound117565.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119337

namespace LeftBound119341
def owner : Owner := ⟨.program ⟨257⟩, ⟨70246⟩⟩
def transferEvent : Nat := 119341
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119339 .coefficient, .predecessor 1 119340 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119339 .coefficient)
      LeftBound119336.bound (LeftBound119336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119336.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119340 .coefficient)
      LeftBound117351.bound (LeftBound117351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events458.exact117358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound117351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound117351.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119336.bound, LeftBound117351.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119336.bound, LeftBound117351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119336.actual selector witness, LeftBound117351.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119341

namespace LeftBound119342
def owner : Owner := ⟨.program ⟨257⟩, ⟨70246⟩⟩
def transferEvent : Nat := 119342
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119338 .summary, .result 117358 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119338 .summary)
      LeftBound119337.bound (LeftBound119337.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64901⟩⟩) (rawTerms := some (Proof.Events466.exact119338RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 117358 .summary)
      LeftBound117353.bound (LeftBound117353.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70245⟩⟩) (rawTerms := some (Proof.Events458.exact117358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound117353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119337.bound, LeftBound117353.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119337.bound, LeftBound117353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119337.actual selector witness, LeftBound117353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119342

namespace LeftBound119346
def owner : Owner := ⟨.program ⟨257⟩, ⟨70247⟩⟩
def transferEvent : Nat := 119346
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119344 .coefficient, .predecessor 1 119345 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119344 .coefficient)
      LeftBound119341.bound (LeftBound119341.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119343RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119341.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119345 .coefficient)
      LeftBound117139.bound (LeftBound117139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events457.exact117146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound117139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound117139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119341.bound, LeftBound117139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119341.bound, LeftBound117139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119341.actual selector witness, LeftBound117139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119346

namespace LeftBound119347
def owner : Owner := ⟨.program ⟨257⟩, ⟨70247⟩⟩
def transferEvent : Nat := 119347
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119343 .summary, .result 117146 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119343 .summary)
      LeftBound119342.bound (LeftBound119342.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70246⟩⟩) (rawTerms := some (Proof.Events466.exact119343RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 117146 .summary)
      LeftBound117141.bound (LeftBound117141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28312⟩⟩) (rawTerms := some (Proof.Events457.exact117146RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound117141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119342.bound, LeftBound117141.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119342.bound, LeftBound117141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119342.actual selector witness, LeftBound117141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119347

namespace LeftBound119351
def owner : Owner := ⟨.program ⟨257⟩, ⟨70248⟩⟩
def transferEvent : Nat := 119351
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119349 .coefficient, .predecessor 1 119350 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119349 .coefficient)
      LeftBound119346.bound (LeftBound119346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119348RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119346.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119350 .coefficient)
      LeftBound116927.bound (LeftBound116927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events456.exact116934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound116927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound116927.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119346.bound, LeftBound116927.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119346.bound, LeftBound116927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119346.actual selector witness, LeftBound116927.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119351

namespace LeftBound119352
def owner : Owner := ⟨.program ⟨257⟩, ⟨70248⟩⟩
def transferEvent : Nat := 119352
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119348 .summary, .result 116934 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119348 .summary)
      LeftBound119347.bound (LeftBound119347.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70247⟩⟩) (rawTerms := some (Proof.Events466.exact119348RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119347.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 116934 .summary)
      LeftBound116929.bound (LeftBound116929.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30992⟩⟩) (rawTerms := some (Proof.Events456.exact116934RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound116929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119347.bound, LeftBound116929.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119347.bound, LeftBound116929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119347.actual selector witness, LeftBound116929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119352

namespace LeftBound119356
def owner : Owner := ⟨.program ⟨257⟩, ⟨70249⟩⟩
def transferEvent : Nat := 119356
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119354 .coefficient, .predecessor 1 119355 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119354 .coefficient)
      LeftBound119351.bound (LeftBound119351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119353RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119351.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119351.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119355 .coefficient)
      LeftBound116715.bound (LeftBound116715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events455.exact116722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound116715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound116715.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119351.bound, LeftBound116715.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119351.bound, LeftBound116715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119351.actual selector witness, LeftBound116715.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119356

namespace LeftBound119357
def owner : Owner := ⟨.program ⟨257⟩, ⟨70249⟩⟩
def transferEvent : Nat := 119357
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119353 .summary, .result 116722 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119353 .summary)
      LeftBound119352.bound (LeftBound119352.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70248⟩⟩) (rawTerms := some (Proof.Events466.exact119353RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 116722 .summary)
      LeftBound116717.bound (LeftBound116717.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36652⟩⟩) (rawTerms := some (Proof.Events455.exact116722RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound116717.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119352.bound, LeftBound116717.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119352.bound, LeftBound116717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119352.actual selector witness, LeftBound116717.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119357

namespace LeftBound119361
def owner : Owner := ⟨.program ⟨257⟩, ⟨70250⟩⟩
def transferEvent : Nat := 119361
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119359 .coefficient, .predecessor 1 119360 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119359 .coefficient)
      LeftBound119356.bound (LeftBound119356.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119358RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119356.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119356.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119360 .coefficient)
      LeftBound116503.bound (LeftBound116503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events455.exact116510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound116503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound116503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119356.bound, LeftBound116503.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119356.bound, LeftBound116503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119356.actual selector witness, LeftBound116503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119361

namespace LeftBound119362
def owner : Owner := ⟨.program ⟨257⟩, ⟨70250⟩⟩
def transferEvent : Nat := 119362
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 119358 .summary, .result 116510 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 119358 .summary)
      LeftBound119357.bound (LeftBound119357.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70249⟩⟩) (rawTerms := some (Proof.Events466.exact119358RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound119357.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 116510 .summary)
      LeftBound116505.bound (LeftBound116505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39332⟩⟩) (rawTerms := some (Proof.Events455.exact116510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound116505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119357.bound, LeftBound116505.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119357.bound, LeftBound116505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119357.actual selector witness, LeftBound116505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119362

namespace LeftBound119366
def owner : Owner := ⟨.program ⟨257⟩, ⟨70251⟩⟩
def transferEvent : Nat := 119366
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 119364 .coefficient, .predecessor 1 119365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 119364 .coefficient)
      LeftBound119361.bound (LeftBound119361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events466.exact119363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound119361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound119361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 119365 .coefficient)
      LeftBound116291.bound (LeftBound116291.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events454.exact116298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound116291.bound, RecordedBoundRefines] <;> decide)
      (LeftBound116291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound119361.bound, LeftBound116291.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound119361.bound, LeftBound116291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound119361.actual selector witness, LeftBound116291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound119366

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
