import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1357

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound202789
def owner : Owner := ⟨.program ⟨257⟩, ⟨54180⟩⟩
def transferEvent : Nat := 202789
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202787 .coefficient, .predecessor 1 202788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202787 .coefficient)
      LeftBound202785.bound (LeftBound202785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202785.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202788 .coefficient)
      LeftAuthority202654.bound (LeftAuthority202654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202654.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202785.bound, LeftAuthority202654.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202785.bound, LeftAuthority202654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202785.actual selector witness, LeftAuthority202654.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202789

namespace LeftBound202793
def owner : Owner := ⟨.program ⟨257⟩, ⟨57160⟩⟩
def transferEvent : Nat := 202793
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202791 .coefficient, .predecessor 1 202792 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202791 .coefficient)
      LeftBound202789.bound (LeftBound202789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202792 .coefficient)
      LeftAuthority202631.bound (LeftAuthority202631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202632RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202631.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202789.bound, LeftAuthority202631.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202789.bound, LeftAuthority202631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202789.actual selector witness, LeftAuthority202631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202793

namespace LeftBound202797
def owner : Owner := ⟨.program ⟨257⟩, ⟨60140⟩⟩
def transferEvent : Nat := 202797
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202795 .coefficient, .predecessor 1 202796 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202795 .coefficient)
      LeftBound202793.bound (LeftBound202793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202794RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202796 .coefficient)
      LeftAuthority202608.bound (LeftAuthority202608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202608.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202793.bound, LeftAuthority202608.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202793.bound, LeftAuthority202608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202793.actual selector witness, LeftAuthority202608.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202797

namespace LeftBound202801
def owner : Owner := ⟨.program ⟨257⟩, ⟨63120⟩⟩
def transferEvent : Nat := 202801
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202799 .coefficient, .predecessor 1 202800 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202799 .coefficient)
      LeftBound202797.bound (LeftBound202797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202800 .coefficient)
      LeftAuthority202585.bound (LeftAuthority202585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202797.bound, LeftAuthority202585.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202797.bound, LeftAuthority202585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202797.actual selector witness, LeftAuthority202585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202801

namespace LeftBound202805
def owner : Owner := ⟨.program ⟨257⟩, ⟨66742⟩⟩
def transferEvent : Nat := 202805
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202803 .coefficient, .predecessor 1 202804 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202803 .coefficient)
      LeftBound202801.bound (LeftBound202801.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202801.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202801.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202804 .coefficient)
      LeftAuthority202562.bound (LeftAuthority202562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202801.bound, LeftAuthority202562.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202801.bound, LeftAuthority202562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202801.actual selector witness, LeftAuthority202562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202805

namespace LeftBound202809
def owner : Owner := ⟨.program ⟨257⟩, ⟨66743⟩⟩
def transferEvent : Nat := 202809
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202807 .coefficient, .predecessor 1 202808 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202807 .coefficient)
      LeftBound202805.bound (LeftBound202805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202805.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202808 .coefficient)
      LeftAuthority202539.bound (LeftAuthority202539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202805.bound, LeftAuthority202539.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202805.bound, LeftAuthority202539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202805.actual selector witness, LeftAuthority202539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202809

namespace LeftBound202813
def owner : Owner := ⟨.program ⟨257⟩, ⟨66744⟩⟩
def transferEvent : Nat := 202813
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202811 .coefficient, .predecessor 1 202812 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202811 .coefficient)
      LeftBound202809.bound (LeftBound202809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202812 .coefficient)
      LeftAuthority202516.bound (LeftAuthority202516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events791.exact202517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202516.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202809.bound, LeftAuthority202516.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202809.bound, LeftAuthority202516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202809.actual selector witness, LeftAuthority202516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202813

namespace LeftBound202817
def owner : Owner := ⟨.program ⟨257⟩, ⟨66745⟩⟩
def transferEvent : Nat := 202817
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202815 .coefficient, .predecessor 1 202816 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202815 .coefficient)
      LeftBound202813.bound (LeftBound202813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202816 .coefficient)
      LeftAuthority202493.bound (LeftAuthority202493.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202493.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202493.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202813.bound, LeftAuthority202493.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202813.bound, LeftAuthority202493.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202813.actual selector witness, LeftAuthority202493.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202817

namespace LeftBound202821
def owner : Owner := ⟨.program ⟨257⟩, ⟨66746⟩⟩
def transferEvent : Nat := 202821
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202819 .coefficient, .predecessor 1 202820 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202819 .coefficient)
      LeftBound202817.bound (LeftBound202817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202820 .coefficient)
      LeftAuthority202470.bound (LeftAuthority202470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202470.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202817.bound, LeftAuthority202470.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202817.bound, LeftAuthority202470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202817.actual selector witness, LeftAuthority202470.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202821

namespace LeftBound202825
def owner : Owner := ⟨.program ⟨257⟩, ⟨66747⟩⟩
def transferEvent : Nat := 202825
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202823 .coefficient, .predecessor 1 202824 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202823 .coefficient)
      LeftBound202821.bound (LeftBound202821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202821.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202824 .coefficient)
      LeftAuthority202447.bound (LeftAuthority202447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202448RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202821.bound, LeftAuthority202447.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202821.bound, LeftAuthority202447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202821.actual selector witness, LeftAuthority202447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202825

namespace LeftBound202829
def owner : Owner := ⟨.program ⟨257⟩, ⟨66748⟩⟩
def transferEvent : Nat := 202829
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202827 .coefficient, .predecessor 1 202828 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202827 .coefficient)
      LeftBound202825.bound (LeftBound202825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202828 .coefficient)
      LeftAuthority202424.bound (LeftAuthority202424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202825.bound, LeftAuthority202424.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202825.bound, LeftAuthority202424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202825.actual selector witness, LeftAuthority202424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202829

namespace LeftBound202833
def owner : Owner := ⟨.program ⟨257⟩, ⟨66749⟩⟩
def transferEvent : Nat := 202833
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202831 .coefficient, .predecessor 1 202832 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202831 .coefficient)
      LeftBound202829.bound (LeftBound202829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202829.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202832 .coefficient)
      LeftAuthority202401.bound (LeftAuthority202401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202829.bound, LeftAuthority202401.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202829.bound, LeftAuthority202401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202829.actual selector witness, LeftAuthority202401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202833

namespace LeftBound202837
def owner : Owner := ⟨.program ⟨257⟩, ⟨66750⟩⟩
def transferEvent : Nat := 202837
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202835 .coefficient, .predecessor 1 202836 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202835 .coefficient)
      LeftBound202833.bound (LeftBound202833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202836 .coefficient)
      LeftAuthority202378.bound (LeftAuthority202378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events790.exact202379RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202378.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202833.bound, LeftAuthority202378.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202833.bound, LeftAuthority202378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202833.actual selector witness, LeftAuthority202378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202837

namespace LeftBound202840
def owner : Owner := ⟨.program ⟨257⟩, ⟨66751⟩⟩
def transferEvent : Nat := 202840
def frameStart : Nat := 202336
def rule : BoundRule := .identity (.predecessor 0 202839 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202839 .coefficient)
      LeftBound202837.bound (LeftBound202837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202837.derived selector witness)

def rawBound : CoeffClass := LeftBound202837.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound202837.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound202840

namespace LeftBound202857
def owner : Owner := ⟨.program ⟨257⟩, ⟨69095⟩⟩
def transferEvent : Nat := 202857
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202855 .coefficient, .predecessor 1 202856 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202855 .coefficient)
      LeftBound202840.bound (LeftBound202840.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound202840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202856 .coefficient)
      LeftAuthority202853.bound (LeftAuthority202853.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority202853.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202840.bound, LeftAuthority202853.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202840.bound, LeftAuthority202853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202840.actual selector witness, LeftAuthority202853.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202857

namespace LeftBound202860
def owner : Owner := ⟨.program ⟨257⟩, ⟨69096⟩⟩
def transferEvent : Nat := 202860
def frameStart : Nat := 202336
def rule : BoundRule := .identity (.predecessor 0 202859 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202859 .coefficient)
      LeftBound202857.bound (LeftBound202857.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound202857.derived selector witness)

def rawBound : CoeffClass := LeftBound202857.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound202857.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound202860

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
