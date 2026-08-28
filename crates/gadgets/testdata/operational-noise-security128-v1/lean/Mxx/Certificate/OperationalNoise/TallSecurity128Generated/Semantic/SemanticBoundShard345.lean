import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard344

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56712
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 56712
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56710 .coefficient, .predecessor 1 56711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56710 .coefficient)
      LeftBound56708.bound (LeftBound56708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56711 .coefficient)
      LeftAuthority56670.bound (LeftAuthority56670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56670.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56670.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56708.bound, LeftAuthority56670.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56708.bound, LeftAuthority56670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56708.actual selector witness, LeftAuthority56670.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56712

namespace LeftBound56716
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 56716
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56714 .coefficient, .predecessor 1 56715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56714 .coefficient)
      LeftBound56712.bound (LeftBound56712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56715 .coefficient)
      LeftAuthority56667.bound (LeftAuthority56667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56712.bound, LeftAuthority56667.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56712.bound, LeftAuthority56667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56712.actual selector witness, LeftAuthority56667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56716

namespace LeftBound56720
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 56720
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56718 .coefficient, .predecessor 1 56719 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56718 .coefficient)
      LeftBound56716.bound (LeftBound56716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56716.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56719 .coefficient)
      LeftAuthority56664.bound (LeftAuthority56664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56716.bound, LeftAuthority56664.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56716.bound, LeftAuthority56664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56716.actual selector witness, LeftAuthority56664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56720

namespace LeftBound56724
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 56724
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56722 .coefficient, .predecessor 1 56723 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56722 .coefficient)
      LeftBound56720.bound (LeftBound56720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56723 .coefficient)
      LeftAuthority56661.bound (LeftAuthority56661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56661.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56661.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56720.bound, LeftAuthority56661.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56720.bound, LeftAuthority56661.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56720.actual selector witness, LeftAuthority56661.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56724

namespace LeftBound56728
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 56728
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56726 .coefficient, .predecessor 1 56727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56726 .coefficient)
      LeftBound56724.bound (LeftBound56724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56727 .coefficient)
      LeftAuthority56658.bound (LeftAuthority56658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56724.bound, LeftAuthority56658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56724.bound, LeftAuthority56658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56724.actual selector witness, LeftAuthority56658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56728

namespace LeftBound56732
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 56732
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56730 .coefficient, .predecessor 1 56731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56730 .coefficient)
      LeftBound56728.bound (LeftBound56728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56731 .coefficient)
      LeftAuthority56655.bound (LeftAuthority56655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56655.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56655.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56728.bound, LeftAuthority56655.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56728.bound, LeftAuthority56655.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56728.actual selector witness, LeftAuthority56655.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56732

namespace LeftBound56736
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 56736
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56734 .coefficient, .predecessor 1 56735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56734 .coefficient)
      LeftBound56732.bound (LeftBound56732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56735 .coefficient)
      LeftAuthority56652.bound (LeftAuthority56652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56652.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56732.bound, LeftAuthority56652.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56732.bound, LeftAuthority56652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56732.actual selector witness, LeftAuthority56652.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56736

namespace LeftBound56740
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 56740
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56738 .coefficient, .predecessor 1 56739 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56738 .coefficient)
      LeftBound56736.bound (LeftBound56736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56739 .coefficient)
      LeftAuthority56649.bound (LeftAuthority56649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56650RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56649.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56649.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56736.bound, LeftAuthority56649.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56736.bound, LeftAuthority56649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56736.actual selector witness, LeftAuthority56649.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56740

namespace LeftBound56744
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 56744
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56742 .coefficient, .predecessor 1 56743 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56742 .coefficient)
      LeftBound56740.bound (LeftBound56740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56740.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56740.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56743 .coefficient)
      LeftAuthority56646.bound (LeftAuthority56646.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56646.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56740.bound, LeftAuthority56646.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56740.bound, LeftAuthority56646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56740.actual selector witness, LeftAuthority56646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56744

namespace LeftBound56748
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 56748
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56746 .coefficient, .predecessor 1 56747 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56746 .coefficient)
      LeftBound56744.bound (LeftBound56744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56747 .coefficient)
      LeftAuthority56643.bound (LeftAuthority56643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56643.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56744.bound, LeftAuthority56643.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56744.bound, LeftAuthority56643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56744.actual selector witness, LeftAuthority56643.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56748

namespace LeftBound56752
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 56752
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56750 .coefficient, .predecessor 1 56751 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56750 .coefficient)
      LeftBound56748.bound (LeftBound56748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56751 .coefficient)
      LeftAuthority56640.bound (LeftAuthority56640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56641RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56640.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56748.bound, LeftAuthority56640.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56748.bound, LeftAuthority56640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56748.actual selector witness, LeftAuthority56640.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56752

namespace LeftBound56756
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 56756
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56754 .coefficient, .predecessor 1 56755 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56754 .coefficient)
      LeftBound56752.bound (LeftBound56752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56755 .coefficient)
      LeftAuthority56637.bound (LeftAuthority56637.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56637.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56637.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56752.bound, LeftAuthority56637.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56752.bound, LeftAuthority56637.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56752.actual selector witness, LeftAuthority56637.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56756

namespace LeftBound56760
def owner : Owner := ⟨.program ⟨257⟩, ⟨69122⟩⟩
def transferEvent : Nat := 56760
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56758 .coefficient, .predecessor 1 56759 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56758 .coefficient)
      LeftBound56756.bound (LeftBound56756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56759 .coefficient)
      LeftBound56616.bound (LeftBound56616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56616.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56756.bound, LeftBound56616.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56756.bound, LeftBound56616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56756.actual selector witness, LeftBound56616.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56760

namespace LeftBound56764
def owner : Owner := ⟨.program ⟨257⟩, ⟨71502⟩⟩
def transferEvent : Nat := 56764
def frameStart : Nat := 56086
def rule : BoundRule := .product (.predecessor 0 56762 .coefficient) (.predecessor 1 56763 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56762 .coefficient)
      LeftBound56760.bound (LeftBound56760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56760.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56763 .coefficient)
      LeftAuthority56601.bound (LeftAuthority56601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56601.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound56760.bound LeftAuthority56601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56760.bound, LeftAuthority56601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound56760.actual selector witness) * (LeftAuthority56601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56764

namespace LeftBound56843
def owner : Owner := ⟨.program ⟨257⟩, ⟨67628⟩⟩
def transferEvent : Nat := 56843
def frameStart : Nat := 56086
def rule : BoundRule := .product (.predecessor 0 56841 .coefficient) (.predecessor 1 56842 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56841 .coefficient)
      LeftAuthority56612.bound (LeftAuthority56612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56842 .coefficient)
      LeftAuthority56839.bound (LeftAuthority56839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56612.bound LeftAuthority56839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56612.bound, LeftAuthority56839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority56612.actual selector witness) * (LeftAuthority56839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56843

namespace LeftBound56851
def owner : Owner := ⟨.program ⟨257⟩, ⟨67633⟩⟩
def transferEvent : Nat := 56851
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56849 .coefficient, .predecessor 1 56850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56849 .coefficient)
      LeftAuthority56847.bound (LeftAuthority56847.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56847.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56850 .coefficient)
      LeftBound56843.bound (LeftBound56843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56843.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56847.bound, LeftBound56843.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56847.bound, LeftBound56843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority56847.actual selector witness, LeftBound56843.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56851

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
