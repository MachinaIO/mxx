import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard118
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1388
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1437

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound213733
def owner : Owner := ⟨.program ⟨257⟩, ⟨55348⟩⟩
def transferEvent : Nat := 213733
def frameStart : Nat := 213668
def rule : BoundRule := .product (.predecessor 0 213731 .coefficient) (.predecessor 1 213732 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213731 .coefficient)
      LeftAuthority213729.bound (LeftAuthority213729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213732 .coefficient)
      LeftBound213727.bound (LeftBound213727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213727.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority213729.bound LeftBound213727.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213729.bound, LeftBound213727.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority213729.actual selector witness) * (LeftBound213727.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213733

namespace LeftBound213741
def owner : Owner := ⟨.program ⟨257⟩, ⟨55349⟩⟩
def transferEvent : Nat := 213741
def frameStart : Nat := 213668
def rule : BoundRule := .sum [.predecessor 0 213739 .coefficient, .predecessor 1 213740 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213739 .coefficient)
      LeftAuthority213737.bound (LeftAuthority213737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213737.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213737.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213740 .coefficient)
      LeftBound213733.bound (LeftBound213733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213735RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority213737.bound, LeftBound213733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213737.bound, LeftBound213733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority213737.actual selector witness, LeftBound213733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213741

namespace LeftBound213745
def owner : Owner := ⟨.program ⟨257⟩, ⟨55933⟩⟩
def transferEvent : Nat := 213745
def frameStart : Nat := 213668
def rule : BoundRule := .product (.predecessor 0 213743 .coefficient) (.predecessor 1 213744 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213743 .coefficient)
      LeftBound213741.bound (LeftBound213741.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213741.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213741.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213744 .coefficient)
      LeftAuthority213718.bound (LeftAuthority213718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound213741.bound LeftAuthority213718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213741.bound, LeftAuthority213718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound213741.actual selector witness) * (LeftAuthority213718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213745

namespace LeftBound213756
def owner : Owner := ⟨.program ⟨257⟩, ⟨54143⟩⟩
def transferEvent : Nat := 213756
def frameStart : Nat := 213668
def rule : BoundRule := .product (.predecessor 0 213754 .coefficient) (.predecessor 1 213755 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213754 .coefficient)
      LeftAuthority213729.bound (LeftAuthority213729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213755 .coefficient)
      LeftAuthority213752.bound (LeftAuthority213752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213752.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority213729.bound LeftAuthority213752.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213729.bound, LeftAuthority213752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority213729.actual selector witness) * (LeftAuthority213752.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213756

namespace LeftBound213764
def owner : Owner := ⟨.program ⟨257⟩, ⟨54144⟩⟩
def transferEvent : Nat := 213764
def frameStart : Nat := 213668
def rule : BoundRule := .sum [.predecessor 0 213762 .coefficient, .predecessor 1 213763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213762 .coefficient)
      LeftAuthority213760.bound (LeftAuthority213760.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority213760.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority213760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213763 .coefficient)
      LeftBound213756.bound (LeftBound213756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213756.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority213760.bound, LeftBound213756.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority213760.bound, LeftBound213756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority213760.actual selector witness, LeftBound213756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213764

namespace LeftBound213768
def owner : Owner := ⟨.program ⟨257⟩, ⟨55937⟩⟩
def transferEvent : Nat := 213768
def frameStart : Nat := 213668
def rule : BoundRule := .sum [.predecessor 0 213766 .coefficient, .predecessor 1 213767 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213766 .coefficient)
      LeftBound213764.bound (LeftBound213764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213767 .coefficient)
      LeftBound213745.bound (LeftBound213745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213764.bound, LeftBound213745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213764.bound, LeftBound213745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213764.actual selector witness, LeftBound213745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213768

namespace LeftBound213781
def owner : Owner := ⟨.program ⟨257⟩, ⟨55935⟩⟩
def transferEvent : Nat := 213781
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 213779 .coefficient, .predecessor 1 213780 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213779 .coefficient)
      LeftBound213610.bound (LeftBound213610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213610.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213780 .coefficient)
      LeftBound213593.bound (LeftBound213593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events834.exact213600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213593.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213593.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213610.bound, LeftBound213593.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213610.bound, LeftBound213593.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213610.actual selector witness, LeftBound213593.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213781

namespace LeftBound213784
def owner : Owner := ⟨.program ⟨257⟩, ⟨55935⟩⟩
def transferEvent : Nat := 213784
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 213778 .summary, .result 213600 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213778 .summary)
      LeftBound213612.bound (LeftBound213612.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54739⟩⟩) (rawTerms := some (Proof.Events835.exact213778RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213600 .summary)
      LeftBound213595.bound (LeftBound213595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55934⟩⟩) (rawTerms := some (Proof.Events834.exact213600RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213612.bound, LeftBound213595.bound]
def bound : CoeffClass := .finite ⟨32189789464712143775715074244608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213612.bound, LeftBound213595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213612.actual selector witness, LeftBound213595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213784

namespace LeftBound213808
def owner : Owner := ⟨.program ⟨257⟩, ⟨24531⟩⟩
def transferEvent : Nat := 213808
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 213806 .coefficient) (.predecessor 1 213807 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213806 .coefficient)
      LeftAuthority10116.bound (LeftAuthority10116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213807 .coefficient)
      LeftBound207526.bound (LeftBound207526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10116.bound LeftBound207526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10116.bound, LeftBound207526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10116.actual selector witness) * (LeftBound207526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound213808

namespace LeftBound213813
def owner : Owner := ⟨.program ⟨257⟩, ⟨8614⟩⟩
def transferEvent : Nat := 213813
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 213811 .coefficient) (.predecessor 1 213812 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213811 .coefficient)
      LeftBound207397.bound (LeftBound207397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events810.exact207398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213812 .coefficient)
      LeftBound23592.bound (LeftBound23592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23592.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound207397.bound LeftBound23592.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207397.bound, LeftBound23592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound207397.actual selector witness) * (LeftBound23592.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213813

namespace LeftBound213818
def owner : Owner := ⟨.program ⟨257⟩, ⟨24532⟩⟩
def transferEvent : Nat := 213818
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 213816 .coefficient, .predecessor 1 213817 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213816 .coefficient)
      LeftBound213813.bound (LeftBound213813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213817 .coefficient)
      LeftBound213808.bound (LeftBound213808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213813.bound, LeftBound213808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213813.bound, LeftBound213808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213813.actual selector witness, LeftBound213808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213818

namespace LeftBound213822
def owner : Owner := ⟨.program ⟨257⟩, ⟨24533⟩⟩
def transferEvent : Nat := 213822
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 213820 .coefficient, .predecessor 1 213821 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213820 .coefficient)
      LeftBound213818.bound (LeftBound213818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213818.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213821 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound213818.bound, LeftBound23584.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213818.bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound213818.actual selector witness, LeftBound23584.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound213822

namespace LeftBound213823
def owner : Owner := ⟨.program ⟨257⟩, ⟨24533⟩⟩
def transferEvent : Nat := 213823
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨134⟩⟩]⟩ [⟨.result 23585 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 23585 .coefficient)
      LeftBound23584.bound (LeftBound23584.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨134⟩⟩) (rawTerms := some (Proof.Events092.exact23585RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23584.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23584.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound23584.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23584.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound23584.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound213823

namespace LeftBound213828
def owner : Owner := ⟨.program ⟨257⟩, ⟨50548⟩⟩
def transferEvent : Nat := 213828
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 213826 .coefficient) (.predecessor 1 213827 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 213826 .coefficient)
      LeftBound213822.bound (LeftBound213822.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events835.exact213825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound213822.bound, RecordedBoundRefines] <;> decide)
      (LeftBound213822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 213827 .coefficient)
      LeftAuthority10119.bound (LeftAuthority10119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10119.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound213822.bound LeftAuthority10119.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213822.bound, LeftAuthority10119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound213822.actual selector witness) * (LeftAuthority10119.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213828

namespace LeftBound213829
def owner : Owner := ⟨.program ⟨257⟩, ⟨50548⟩⟩
def transferEvent : Nat := 213829
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨50545⟩⟩], []⟩ [⟨.result 10120 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 10120 .coefficient)
      LeftAuthority10119.bound (LeftAuthority10119.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨50545⟩⟩) (rawTerms := some (Proof.Events039.exact10120RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10119.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10119.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority10119.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound213829

namespace LeftBound213830
def owner : Owner := ⟨.program ⟨257⟩, ⟨50548⟩⟩
def transferEvent : Nat := 213830
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 213825 .summary) (.transfer 213829) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 213825 .summary)
      LeftBound213823.bound (LeftBound213823.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24533⟩⟩) (rawTerms := some (Proof.Events835.exact213825RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound213823.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 213829)
      LeftBound213829.bound (LeftBound213829.actual selector witness) := by
  exact .transfer (LeftBound213829.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound213823.bound LeftBound213829.bound
def bound : CoeffClass := .finite ⟨8519680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound213823.bound, LeftBound213829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound213823.actual selector witness) * (LeftBound213829.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound213830

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
