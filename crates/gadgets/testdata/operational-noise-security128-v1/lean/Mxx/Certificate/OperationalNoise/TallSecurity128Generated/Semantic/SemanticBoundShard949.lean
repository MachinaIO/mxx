import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard931
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard934
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard938
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard942
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard945
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard948

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound143026
def owner : Owner := ⟨.program ⟨257⟩, ⟨17181⟩⟩
def transferEvent : Nat := 143026
def frameStart : Nat := 142953
def rule : BoundRule := .sum [.predecessor 0 143024 .coefficient, .predecessor 1 143025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143024 .coefficient)
      LeftAuthority143022.bound (LeftAuthority143022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143025 .coefficient)
      LeftBound143018.bound (LeftBound143018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143018.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority143022.bound, LeftBound143018.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority143022.bound, LeftBound143018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority143022.actual selector witness, LeftBound143018.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143026

namespace LeftBound143030
def owner : Owner := ⟨.program ⟨257⟩, ⟨17566⟩⟩
def transferEvent : Nat := 143030
def frameStart : Nat := 142953
def rule : BoundRule := .product (.predecessor 0 143028 .coefficient) (.predecessor 1 143029 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143028 .coefficient)
      LeftBound143026.bound (LeftBound143026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143029 .coefficient)
      LeftAuthority143003.bound (LeftAuthority143003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound143026.bound LeftAuthority143003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143026.bound, LeftAuthority143003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound143026.actual selector witness) * (LeftAuthority143003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143030

namespace LeftBound143041
def owner : Owner := ⟨.program ⟨257⟩, ⟨15924⟩⟩
def transferEvent : Nat := 143041
def frameStart : Nat := 142953
def rule : BoundRule := .product (.predecessor 0 143039 .coefficient) (.predecessor 1 143040 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143039 .coefficient)
      LeftAuthority143014.bound (LeftAuthority143014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143040 .coefficient)
      LeftAuthority143037.bound (LeftAuthority143037.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143037.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority143014.bound LeftAuthority143037.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority143014.bound, LeftAuthority143037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority143014.actual selector witness) * (LeftAuthority143037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound143041

namespace LeftBound143049
def owner : Owner := ⟨.program ⟨257⟩, ⟨15925⟩⟩
def transferEvent : Nat := 143049
def frameStart : Nat := 142953
def rule : BoundRule := .sum [.predecessor 0 143047 .coefficient, .predecessor 1 143048 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143047 .coefficient)
      LeftAuthority143045.bound (LeftAuthority143045.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143046RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority143045.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority143045.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143048 .coefficient)
      LeftBound143041.bound (LeftBound143041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority143045.bound, LeftBound143041.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority143045.bound, LeftBound143041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority143045.actual selector witness, LeftBound143041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143049

namespace LeftBound143053
def owner : Owner := ⟨.program ⟨257⟩, ⟨17569⟩⟩
def transferEvent : Nat := 143053
def frameStart : Nat := 142953
def rule : BoundRule := .sum [.predecessor 0 143051 .coefficient, .predecessor 1 143052 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143051 .coefficient)
      LeftBound143049.bound (LeftBound143049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143050RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143049.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143052 .coefficient)
      LeftBound143030.bound (LeftBound143030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143030.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143049.bound, LeftBound143030.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143049.bound, LeftBound143030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143049.actual selector witness, LeftBound143030.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143053

namespace LeftBound143066
def owner : Owner := ⟨.program ⟨257⟩, ⟨17568⟩⟩
def transferEvent : Nat := 143066
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143064 .coefficient, .predecessor 1 143065 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143064 .coefficient)
      LeftBound142895.bound (LeftBound142895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143065 .coefficient)
      LeftBound142878.bound (LeftBound142878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact142885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142878.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142895.bound, LeftBound142878.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142895.bound, LeftBound142878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142895.actual selector witness, LeftBound142878.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143066

namespace LeftBound143069
def owner : Owner := ⟨.program ⟨257⟩, ⟨17568⟩⟩
def transferEvent : Nat := 143069
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143063 .summary, .result 142885 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143063 .summary)
      LeftBound142897.bound (LeftBound142897.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16459⟩⟩) (rawTerms := some (Proof.Events558.exact143063RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142885 .summary)
      LeftBound142880.bound (LeftBound142880.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17567⟩⟩) (rawTerms := some (Proof.Events558.exact142885RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142880.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142897.bound, LeftBound142880.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142897.bound, LeftBound142880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142897.actual selector witness, LeftBound142880.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143069

namespace LeftBound143073
def owner : Owner := ⟨.program ⟨257⟩, ⟨20439⟩⟩
def transferEvent : Nat := 143073
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143071 .coefficient, .predecessor 1 143072 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143071 .coefficient)
      LeftBound143066.bound (LeftBound143066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143072 .coefficient)
      LeftBound142584.bound (LeftBound142584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143066.bound, LeftBound142584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143066.bound, LeftBound142584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143066.actual selector witness, LeftBound142584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143073

namespace LeftBound143074
def owner : Owner := ⟨.program ⟨257⟩, ⟨20439⟩⟩
def transferEvent : Nat := 143074
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143070 .summary, .result 142588 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143070 .summary)
      LeftBound143069.bound (LeftBound143069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17568⟩⟩) (rawTerms := some (Proof.Events558.exact143070RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142588 .summary)
      LeftBound142587.bound (LeftBound142587.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20438⟩⟩) (rawTerms := some (Proof.Events556.exact142588RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142587.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143069.bound, LeftBound142587.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143069.bound, LeftBound142587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143069.actual selector witness, LeftBound142587.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143074

namespace LeftBound143078
def owner : Owner := ⟨.program ⟨257⟩, ⟨23659⟩⟩
def transferEvent : Nat := 143078
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143076 .coefficient, .predecessor 1 143077 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143076 .coefficient)
      LeftBound143073.bound (LeftBound143073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143073.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143077 .coefficient)
      LeftBound142102.bound (LeftBound142102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events555.exact142106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143073.bound, LeftBound142102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143073.bound, LeftBound142102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143073.actual selector witness, LeftBound142102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143078

namespace LeftBound143079
def owner : Owner := ⟨.program ⟨257⟩, ⟨23659⟩⟩
def transferEvent : Nat := 143079
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143075 .summary, .result 142106 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143075 .summary)
      LeftBound143074.bound (LeftBound143074.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20439⟩⟩) (rawTerms := some (Proof.Events558.exact143075RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142106 .summary)
      LeftBound142105.bound (LeftBound142105.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23658⟩⟩) (rawTerms := some (Proof.Events555.exact142106RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143074.bound, LeftBound142105.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143074.bound, LeftBound142105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143074.actual selector witness, LeftBound142105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143079

namespace LeftBound143083
def owner : Owner := ⟨.program ⟨257⟩, ⟨33679⟩⟩
def transferEvent : Nat := 143083
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143081 .coefficient, .predecessor 1 143082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143081 .coefficient)
      LeftBound143078.bound (LeftBound143078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143078.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143082 .coefficient)
      LeftBound141620.bound (LeftBound141620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events553.exact141624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143078.bound, LeftBound141620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143078.bound, LeftBound141620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143078.actual selector witness, LeftBound141620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143083

namespace LeftBound143084
def owner : Owner := ⟨.program ⟨257⟩, ⟨33679⟩⟩
def transferEvent : Nat := 143084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143080 .summary, .result 141624 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143080 .summary)
      LeftBound143079.bound (LeftBound143079.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23659⟩⟩) (rawTerms := some (Proof.Events558.exact143080RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 141624 .summary)
      LeftBound141623.bound (LeftBound141623.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33678⟩⟩) (rawTerms := some (Proof.Events553.exact141624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound141623.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143079.bound, LeftBound141623.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143079.bound, LeftBound141623.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143079.actual selector witness, LeftBound141623.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143084

namespace LeftBound143088
def owner : Owner := ⟨.program ⟨257⟩, ⟨52739⟩⟩
def transferEvent : Nat := 143088
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143086 .coefficient, .predecessor 1 143087 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143086 .coefficient)
      LeftBound143083.bound (LeftBound143083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143087 .coefficient)
      LeftBound141138.bound (LeftBound141138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events551.exact141142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound141138.bound, RecordedBoundRefines] <;> decide)
      (LeftBound141138.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143083.bound, LeftBound141138.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143083.bound, LeftBound141138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143083.actual selector witness, LeftBound141138.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143088

namespace LeftBound143089
def owner : Owner := ⟨.program ⟨257⟩, ⟨52739⟩⟩
def transferEvent : Nat := 143089
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 143085 .summary, .result 141142 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 143085 .summary)
      LeftBound143084.bound (LeftBound143084.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33679⟩⟩) (rawTerms := some (Proof.Events558.exact143085RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound143084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 141142 .summary)
      LeftBound141141.bound (LeftBound141141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52738⟩⟩) (rawTerms := some (Proof.Events551.exact141142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound141141.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143084.bound, LeftBound141141.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143084.bound, LeftBound141141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143084.actual selector witness, LeftBound141141.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143089

namespace LeftBound143093
def owner : Owner := ⟨.program ⟨257⟩, ⟨55719⟩⟩
def transferEvent : Nat := 143093
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 143091 .coefficient, .predecessor 1 143092 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 143091 .coefficient)
      LeftBound143088.bound (LeftBound143088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events558.exact143090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound143088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound143088.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 143092 .coefficient)
      LeftBound140656.bound (LeftBound140656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events549.exact140660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound140656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound140656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound143088.bound, LeftBound140656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound143088.bound, LeftBound140656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound143088.actual selector witness, LeftBound140656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound143093

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
