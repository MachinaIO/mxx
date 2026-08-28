import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard662
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard663
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard664
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard665
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard666
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard667
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard668
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard669
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard670
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard671
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard674

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104681
def owner : Owner := ⟨.program ⟨257⟩, ⟨24025⟩⟩
def transferEvent : Nat := 104681
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104679 .coefficient, .predecessor 1 104680 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104679 .coefficient)
      LeftBound104676.bound (LeftBound104676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104678RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104680 .coefficient)
      LeftBound104210.bound (LeftBound104210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104676.bound, LeftBound104210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104676.bound, LeftBound104210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104676.actual selector witness, LeftBound104210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104681

namespace LeftBound104682
def owner : Owner := ⟨.program ⟨257⟩, ⟨24025⟩⟩
def transferEvent : Nat := 104682
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104678 .summary, .result 104217 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104678 .summary)
      LeftBound104677.bound (LeftBound104677.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20805⟩⟩) (rawTerms := some (Proof.Events408.exact104678RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104677.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104217 .summary)
      LeftBound104212.bound (LeftBound104212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24024⟩⟩) (rawTerms := some (Proof.Events407.exact104217RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104212.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104677.bound, LeftBound104212.bound]
def bound : CoeffClass := .finite ⟨1036877221117396499835321299770218916085812, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104677.bound, LeftBound104212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104677.actual selector witness, LeftBound104212.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104682

namespace LeftBound104686
def owner : Owner := ⟨.program ⟨257⟩, ⟨34045⟩⟩
def transferEvent : Nat := 104686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104684 .coefficient, .predecessor 1 104685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104684 .coefficient)
      LeftBound104681.bound (LeftBound104681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104685 .coefficient)
      LeftBound103998.bound (LeftBound103998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103998.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104681.bound, LeftBound103998.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104681.bound, LeftBound103998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104681.actual selector witness, LeftBound103998.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104686

namespace LeftBound104687
def owner : Owner := ⟨.program ⟨257⟩, ⟨34045⟩⟩
def transferEvent : Nat := 104687
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104683 .summary, .result 104005 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104683 .summary)
      LeftBound104682.bound (LeftBound104682.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24025⟩⟩) (rawTerms := some (Proof.Events408.exact104683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104005 .summary)
      LeftBound104000.bound (LeftBound104000.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34044⟩⟩) (rawTerms := some (Proof.Events406.exact104005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104000.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104682.bound, LeftBound104000.bound]
def bound : CoeffClass := .finite ⟨1382506125545760169441014535464825839943732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104682.bound, LeftBound104000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104682.actual selector witness, LeftBound104000.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104687

namespace LeftBound104691
def owner : Owner := ⟨.program ⟨257⟩, ⟨53105⟩⟩
def transferEvent : Nat := 104691
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104689 .coefficient, .predecessor 1 104690 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104689 .coefficient)
      LeftBound104686.bound (LeftBound104686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104690 .coefficient)
      LeftBound103786.bound (LeftBound103786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events405.exact103793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103786.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104686.bound, LeftBound103786.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104686.bound, LeftBound103786.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104686.actual selector witness, LeftBound103786.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104691

namespace LeftBound104692
def owner : Owner := ⟨.program ⟨257⟩, ⟨53105⟩⟩
def transferEvent : Nat := 104692
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104688 .summary, .result 103793 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104688 .summary)
      LeftBound104687.bound (LeftBound104687.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34045⟩⟩) (rawTerms := some (Proof.Events408.exact104688RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103793 .summary)
      LeftBound103788.bound (LeftBound103788.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53104⟩⟩) (rawTerms := some (Proof.Events405.exact103793RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103788.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104687.bound, LeftBound103788.bound]
def bound : CoeffClass := .finite ⟨1728139248715321398594155952187700255129652, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104687.bound, LeftBound103788.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104687.actual selector witness, LeftBound103788.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104692

namespace LeftBound104696
def owner : Owner := ⟨.program ⟨257⟩, ⟨56085⟩⟩
def transferEvent : Nat := 104696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104694 .coefficient, .predecessor 1 104695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104694 .coefficient)
      LeftBound104691.bound (LeftBound104691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104695 .coefficient)
      LeftBound103574.bound (LeftBound103574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events404.exact103581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104691.bound, LeftBound103574.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104691.bound, LeftBound103574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104691.actual selector witness, LeftBound103574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104696

namespace LeftBound104697
def owner : Owner := ⟨.program ⟨257⟩, ⟨56085⟩⟩
def transferEvent : Nat := 104697
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104693 .summary, .result 103581 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104693 .summary)
      LeftBound104692.bound (LeftBound104692.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53105⟩⟩) (rawTerms := some (Proof.Events408.exact104693RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103581 .summary)
      LeftBound103576.bound (LeftBound103576.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56084⟩⟩) (rawTerms := some (Proof.Events404.exact103581RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104692.bound, LeftBound103576.bound]
def bound : CoeffClass := .finite ⟨2073774481255481407521021459424708415979572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104692.bound, LeftBound103576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104692.actual selector witness, LeftBound103576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104697

namespace LeftBound104701
def owner : Owner := ⟨.program ⟨257⟩, ⟨59065⟩⟩
def transferEvent : Nat := 104701
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104699 .coefficient, .predecessor 1 104700 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104699 .coefficient)
      LeftBound104696.bound (LeftBound104696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104700 .coefficient)
      LeftBound103362.bound (LeftBound103362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events403.exact103369RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103362.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103362.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104696.bound, LeftBound103362.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104696.bound, LeftBound103362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104696.actual selector witness, LeftBound103362.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104701

namespace LeftBound104702
def owner : Owner := ⟨.program ⟨257⟩, ⟨59065⟩⟩
def transferEvent : Nat := 104702
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104698 .summary, .result 103369 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104698 .summary)
      LeftBound104697.bound (LeftBound104697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56085⟩⟩) (rawTerms := some (Proof.Events408.exact104698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103369 .summary)
      LeftBound103364.bound (LeftBound103364.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59064⟩⟩) (rawTerms := some (Proof.Events403.exact103369RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103364.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104697.bound, LeftBound103364.bound]
def bound : CoeffClass := .finite ⟨2419413932536838975995335147689984068157492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104697.bound, LeftBound103364.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104697.actual selector witness, LeftBound103364.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104702

namespace LeftBound104706
def owner : Owner := ⟨.program ⟨257⟩, ⟨62045⟩⟩
def transferEvent : Nat := 104706
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104704 .coefficient, .predecessor 1 104705 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104704 .coefficient)
      LeftBound104701.bound (LeftBound104701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events408.exact104703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104701.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104705 .coefficient)
      LeftBound103150.bound (LeftBound103150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact103157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound103150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound103150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104701.bound, LeftBound103150.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104701.bound, LeftBound103150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104701.actual selector witness, LeftBound103150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104706

namespace LeftBound104707
def owner : Owner := ⟨.program ⟨257⟩, ⟨62045⟩⟩
def transferEvent : Nat := 104707
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104703 .summary, .result 103157 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104703 .summary)
      LeftBound104702.bound (LeftBound104702.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59065⟩⟩) (rawTerms := some (Proof.Events408.exact104703RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 103157 .summary)
      LeftBound103152.bound (LeftBound103152.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62044⟩⟩) (rawTerms := some (Proof.Events402.exact103157RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound103152.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104702.bound, LeftBound103152.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104702.bound, LeftBound103152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104702.actual selector witness, LeftBound103152.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104707

namespace LeftBound104711
def owner : Owner := ⟨.program ⟨257⟩, ⟨65025⟩⟩
def transferEvent : Nat := 104711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104709 .coefficient, .predecessor 1 104710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104709 .coefficient)
      LeftBound104706.bound (LeftBound104706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104706.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104710 .coefficient)
      LeftBound102938.bound (LeftBound102938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events402.exact102945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104706.bound, LeftBound102938.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104706.bound, LeftBound102938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104706.actual selector witness, LeftBound102938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104711

namespace LeftBound104712
def owner : Owner := ⟨.program ⟨257⟩, ⟨65025⟩⟩
def transferEvent : Nat := 104712
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104708 .summary, .result 102945 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104708 .summary)
      LeftBound104707.bound (LeftBound104707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62045⟩⟩) (rawTerms := some (Proof.Events409.exact104708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102945 .summary)
      LeftBound102940.bound (LeftBound102940.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65024⟩⟩) (rawTerms := some (Proof.Events402.exact102945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102940.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104707.bound, LeftBound102940.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104707.bound, LeftBound102940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104707.actual selector witness, LeftBound102940.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104712

namespace LeftBound104716
def owner : Owner := ⟨.program ⟨257⟩, ⟨70562⟩⟩
def transferEvent : Nat := 104716
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104714 .coefficient, .predecessor 1 104715 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104714 .coefficient)
      LeftBound104711.bound (LeftBound104711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events409.exact104713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104715 .coefficient)
      LeftBound102726.bound (LeftBound102726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events401.exact102733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound102726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound102726.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104711.bound, LeftBound102726.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104711.bound, LeftBound102726.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104711.actual selector witness, LeftBound102726.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104716

namespace LeftBound104717
def owner : Owner := ⟨.program ⟨257⟩, ⟨70562⟩⟩
def transferEvent : Nat := 104717
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104713 .summary, .result 102733 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104713 .summary)
      LeftBound104712.bound (LeftBound104712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65025⟩⟩) (rawTerms := some (Proof.Events409.exact104713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 102733 .summary)
      LeftBound102728.bound (LeftBound102728.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70561⟩⟩) (rawTerms := some (Proof.Events401.exact102733RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102728.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104712.bound, LeftBound102728.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104712.bound, LeftBound102728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104712.actual selector witness, LeftBound102728.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104717

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
