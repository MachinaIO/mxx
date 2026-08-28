import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard851

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound129825
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 129825
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129823 .coefficient, .predecessor 1 129824 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129823 .coefficient)
      LeftBound129821.bound (LeftBound129821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129824 .coefficient)
      LeftAuthority129804.bound (LeftAuthority129804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129821.bound, LeftAuthority129804.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129821.bound, LeftAuthority129804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129821.actual selector witness, LeftAuthority129804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129825

namespace LeftBound129829
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 129829
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129827 .coefficient, .predecessor 1 129828 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129827 .coefficient)
      LeftBound129825.bound (LeftBound129825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129828 .coefficient)
      LeftAuthority129801.bound (LeftAuthority129801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129801.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129801.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129825.bound, LeftAuthority129801.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129825.bound, LeftAuthority129801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129825.actual selector witness, LeftAuthority129801.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129829

namespace LeftBound129833
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 129833
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129831 .coefficient, .predecessor 1 129832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129831 .coefficient)
      LeftBound129829.bound (LeftBound129829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129832 .coefficient)
      LeftAuthority129798.bound (LeftAuthority129798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129829.bound, LeftAuthority129798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129829.bound, LeftAuthority129798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129829.actual selector witness, LeftAuthority129798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129833

namespace LeftBound129837
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 129837
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129835 .coefficient, .predecessor 1 129836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129835 .coefficient)
      LeftBound129833.bound (LeftBound129833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129836 .coefficient)
      LeftAuthority129795.bound (LeftAuthority129795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129795.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129795.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129833.bound, LeftAuthority129795.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129833.bound, LeftAuthority129795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129833.actual selector witness, LeftAuthority129795.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129837

namespace LeftBound129841
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 129841
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129839 .coefficient, .predecessor 1 129840 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129839 .coefficient)
      LeftBound129837.bound (LeftBound129837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129837.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129840 .coefficient)
      LeftAuthority129792.bound (LeftAuthority129792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129792.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129792.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129837.bound, LeftAuthority129792.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129837.bound, LeftAuthority129792.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129837.actual selector witness, LeftAuthority129792.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129841

namespace LeftBound129845
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 129845
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129843 .coefficient, .predecessor 1 129844 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129843 .coefficient)
      LeftBound129841.bound (LeftBound129841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129841.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129844 .coefficient)
      LeftAuthority129789.bound (LeftAuthority129789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129841.bound, LeftAuthority129789.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129841.bound, LeftAuthority129789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129841.actual selector witness, LeftAuthority129789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129845

namespace LeftBound129849
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 129849
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129847 .coefficient, .predecessor 1 129848 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129847 .coefficient)
      LeftBound129845.bound (LeftBound129845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129845.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129845.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129848 .coefficient)
      LeftAuthority129786.bound (LeftAuthority129786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129787RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129786.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129786.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129845.bound, LeftAuthority129786.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129845.bound, LeftAuthority129786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129845.actual selector witness, LeftAuthority129786.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129849

namespace LeftBound129853
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 129853
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129851 .coefficient, .predecessor 1 129852 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129851 .coefficient)
      LeftBound129849.bound (LeftBound129849.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129850RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129849.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129852 .coefficient)
      LeftAuthority129783.bound (LeftAuthority129783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129783.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129783.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129849.bound, LeftAuthority129783.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129849.bound, LeftAuthority129783.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129849.actual selector witness, LeftAuthority129783.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129853

namespace LeftBound129857
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 129857
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129855 .coefficient, .predecessor 1 129856 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129855 .coefficient)
      LeftBound129853.bound (LeftBound129853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129853.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129853.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129856 .coefficient)
      LeftAuthority129780.bound (LeftAuthority129780.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129780.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129780.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129853.bound, LeftAuthority129780.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129853.bound, LeftAuthority129780.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129853.actual selector witness, LeftAuthority129780.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129857

namespace LeftBound129861
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 129861
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129859 .coefficient, .predecessor 1 129860 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129859 .coefficient)
      LeftBound129857.bound (LeftBound129857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129857.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129860 .coefficient)
      LeftAuthority129777.bound (LeftAuthority129777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129857.bound, LeftAuthority129777.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129857.bound, LeftAuthority129777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129857.actual selector witness, LeftAuthority129777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129861

namespace LeftBound129865
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 129865
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129863 .coefficient, .predecessor 1 129864 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129863 .coefficient)
      LeftBound129861.bound (LeftBound129861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129864 .coefficient)
      LeftAuthority129774.bound (LeftAuthority129774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129774.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129861.bound, LeftAuthority129774.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129861.bound, LeftAuthority129774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129861.actual selector witness, LeftAuthority129774.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129865

namespace LeftBound129869
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 129869
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129867 .coefficient, .predecessor 1 129868 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129867 .coefficient)
      LeftBound129865.bound (LeftBound129865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129865.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129865.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129868 .coefficient)
      LeftAuthority129771.bound (LeftAuthority129771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129771.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129771.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129865.bound, LeftAuthority129771.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129865.bound, LeftAuthority129771.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129865.actual selector witness, LeftAuthority129771.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129869

namespace LeftBound129873
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 129873
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129871 .coefficient, .predecessor 1 129872 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129871 .coefficient)
      LeftBound129869.bound (LeftBound129869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129872 .coefficient)
      LeftAuthority129768.bound (LeftAuthority129768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129769RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129768.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129869.bound, LeftAuthority129768.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129869.bound, LeftAuthority129768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129869.actual selector witness, LeftAuthority129768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129873

namespace LeftBound129877
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 129877
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129875 .coefficient, .predecessor 1 129876 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129875 .coefficient)
      LeftBound129873.bound (LeftBound129873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129874RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129873.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129876 .coefficient)
      LeftAuthority129765.bound (LeftAuthority129765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129765.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129765.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129873.bound, LeftAuthority129765.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129873.bound, LeftAuthority129765.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129873.actual selector witness, LeftAuthority129765.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129877

namespace LeftBound129881
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 129881
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129879 .coefficient, .predecessor 1 129880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129879 .coefficient)
      LeftBound129877.bound (LeftBound129877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129877.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129880 .coefficient)
      LeftAuthority129762.bound (LeftAuthority129762.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129763RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority129762.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority129762.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129877.bound, LeftAuthority129762.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129877.bound, LeftAuthority129762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129877.actual selector witness, LeftAuthority129762.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129881

namespace LeftBound129885
def owner : Owner := ⟨.program ⟨257⟩, ⟨69074⟩⟩
def transferEvent : Nat := 129885
def frameStart : Nat := 129211
def rule : BoundRule := .sum [.predecessor 0 129883 .coefficient, .predecessor 1 129884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 129883 .coefficient)
      LeftBound129881.bound (LeftBound129881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events507.exact129882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 129884 .coefficient)
      LeftBound129741.bound (LeftBound129741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events506.exact129760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound129741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound129741.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound129881.bound, LeftBound129741.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound129881.bound, LeftBound129741.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound129881.actual selector witness, LeftBound129741.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound129885

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
