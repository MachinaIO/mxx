import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard509
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard558

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound87828
def owner : Owner := ⟨.program ⟨257⟩, ⟨27791⟩⟩
def transferEvent : Nat := 87828
def frameStart : Nat := 87769
def rule : BoundRule := .identity (.predecessor 0 87827 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87827 .coefficient)
      LeftBound87825.bound (LeftBound87825.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87825.derived selector witness)

def rawBound : CoeffClass := LeftBound87825.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound87825.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87828

namespace LeftBound87834
def owner : Owner := ⟨.program ⟨257⟩, ⟨27792⟩⟩
def transferEvent : Nat := 87834
def frameStart : Nat := 87769
def rule : BoundRule := .product (.predecessor 0 87832 .coefficient) (.predecessor 1 87833 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87832 .coefficient)
      LeftAuthority87830.bound (LeftAuthority87830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87833 .coefficient)
      LeftBound87828.bound (LeftBound87828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87828.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority87830.bound LeftBound87828.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87830.bound, LeftBound87828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority87830.actual selector witness) * (LeftBound87828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87834

namespace LeftBound87842
def owner : Owner := ⟨.program ⟨257⟩, ⟨27793⟩⟩
def transferEvent : Nat := 87842
def frameStart : Nat := 87769
def rule : BoundRule := .sum [.predecessor 0 87840 .coefficient, .predecessor 1 87841 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87840 .coefficient)
      LeftAuthority87838.bound (LeftAuthority87838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87838.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87841 .coefficient)
      LeftBound87834.bound (LeftBound87834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87838.bound, LeftBound87834.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87838.bound, LeftBound87834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority87838.actual selector witness, LeftBound87834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87842

namespace LeftBound87846
def owner : Owner := ⟨.program ⟨257⟩, ⟨28434⟩⟩
def transferEvent : Nat := 87846
def frameStart : Nat := 87769
def rule : BoundRule := .product (.predecessor 0 87844 .coefficient) (.predecessor 1 87845 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87844 .coefficient)
      LeftBound87842.bound (LeftBound87842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87845 .coefficient)
      LeftAuthority87819.bound (LeftAuthority87819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound87842.bound LeftAuthority87819.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87842.bound, LeftAuthority87819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound87842.actual selector witness) * (LeftAuthority87819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87846

namespace LeftBound87857
def owner : Owner := ⟨.program ⟨257⟩, ⟨26702⟩⟩
def transferEvent : Nat := 87857
def frameStart : Nat := 87769
def rule : BoundRule := .product (.predecessor 0 87855 .coefficient) (.predecessor 1 87856 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87855 .coefficient)
      LeftAuthority87830.bound (LeftAuthority87830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87856 .coefficient)
      LeftAuthority87853.bound (LeftAuthority87853.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87854RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87853.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87853.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87830.bound LeftAuthority87853.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87830.bound, LeftAuthority87853.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority87830.actual selector witness) * (LeftAuthority87853.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87857

namespace LeftBound87865
def owner : Owner := ⟨.program ⟨257⟩, ⟨26703⟩⟩
def transferEvent : Nat := 87865
def frameStart : Nat := 87769
def rule : BoundRule := .sum [.predecessor 0 87863 .coefficient, .predecessor 1 87864 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87863 .coefficient)
      LeftAuthority87861.bound (LeftAuthority87861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87861.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87864 .coefficient)
      LeftBound87857.bound (LeftBound87857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87857.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87861.bound, LeftBound87857.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87861.bound, LeftBound87857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority87861.actual selector witness, LeftBound87857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87865

namespace LeftBound87869
def owner : Owner := ⟨.program ⟨257⟩, ⟨28438⟩⟩
def transferEvent : Nat := 87869
def frameStart : Nat := 87769
def rule : BoundRule := .sum [.predecessor 0 87867 .coefficient, .predecessor 1 87868 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87867 .coefficient)
      LeftBound87865.bound (LeftBound87865.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87865.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87865.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87868 .coefficient)
      LeftBound87846.bound (LeftBound87846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87851RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87846.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87865.bound, LeftBound87846.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87865.bound, LeftBound87846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87865.actual selector witness, LeftBound87846.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87869

namespace LeftBound87882
def owner : Owner := ⟨.program ⟨257⟩, ⟨28436⟩⟩
def transferEvent : Nat := 87882
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87880 .coefficient, .predecessor 1 87881 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87880 .coefficient)
      LeftBound87711.bound (LeftBound87711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87881 .coefficient)
      LeftBound87694.bound (LeftBound87694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events342.exact87701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87711.bound, LeftBound87694.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87711.bound, LeftBound87694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87711.actual selector witness, LeftBound87694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87882

namespace LeftBound87885
def owner : Owner := ⟨.program ⟨257⟩, ⟨28436⟩⟩
def transferEvent : Nat := 87885
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 87879 .summary, .result 87701 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87879 .summary)
      LeftBound87713.bound (LeftBound87713.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27275⟩⟩) (rawTerms := some (Proof.Events343.exact87879RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87713.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87701 .summary)
      LeftBound87696.bound (LeftBound87696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28435⟩⟩) (rawTerms := some (Proof.Events342.exact87701RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87696.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87713.bound, LeftBound87696.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87713.bound, LeftBound87696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound87713.actual selector witness, LeftBound87696.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87885

namespace LeftBound87889
def owner : Owner := ⟨.program ⟨257⟩, ⟨28437⟩⟩
def transferEvent : Nat := 87889
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87887 .coefficient) (.predecessor 1 87888 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87887 .coefficient)
      LeftBound87882.bound (LeftBound87882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87882.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87888 .coefficient)
      LeftBound15681.bound (LeftBound15681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound87882.bound LeftBound15681.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87882.bound, LeftBound15681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound87882.actual selector witness) * (LeftBound15681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87889

namespace LeftBound87890
def owner : Owner := ⟨.program ⟨257⟩, ⟨28437⟩⟩
def transferEvent : Nat := 87890
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7169⟩⟩]⟩ [⟨.result 15678 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15678 .coefficient)
      LeftAuthority15677.bound (LeftAuthority15677.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7169⟩⟩) (rawTerms := some (Proof.Events061.exact15678RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15677.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15677.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15677.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15677.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87890

namespace LeftBound87891
def owner : Owner := ⟨.program ⟨257⟩, ⟨28437⟩⟩
def transferEvent : Nat := 87891
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 87886 .summary) (.transfer 87890) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87886 .summary)
      LeftBound87885.bound (LeftBound87885.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28436⟩⟩) (rawTerms := some (Proof.Events343.exact87886RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound87885.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 87890)
      LeftBound87890.bound (LeftBound87890.actual selector witness) := by
  exact .transfer (LeftBound87890.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound87885.bound LeftBound87890.bound
def bound : CoeffClass := .finite ⟨345654216875549026890382321864211871825920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87885.bound, LeftBound87890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound87885.actual selector witness) * (LeftBound87890.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87891

namespace LeftBound87906
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def transferEvent : Nat := 87906
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 87904 .coefficient) (.predecessor 1 87905 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87904 .coefficient)
      LeftBound80033.bound (LeftBound80033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87905 .coefficient)
      LeftAuthority87902.bound (LeftAuthority87902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87902.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound80033.bound LeftAuthority87902.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80033.bound, LeftAuthority87902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound80033.actual selector witness) * (LeftAuthority87902.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87906

namespace LeftBound87907
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def transferEvent : Nat := 87907
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70636⟩⟩]⟩ [⟨.result 87903 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 87903 .coefficient)
      LeftAuthority87902.bound (LeftAuthority87902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70636⟩⟩) (rawTerms := some (Proof.Events343.exact87903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority87902.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority87902.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound87907

namespace LeftBound87908
def owner : Owner := ⟨.program ⟨257⟩, ⟨70638⟩⟩
def transferEvent : Nat := 87908
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80037 .summary) (.transfer 87907) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 80037 .summary)
      LeftBound80036.bound (LeftBound80036.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69308⟩⟩) (rawTerms := some (Proof.Events312.exact80037RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 87907)
      LeftBound87907.bound (LeftBound87907.actual selector witness) := by
  exact .transfer (LeftBound87907.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound80036.bound LeftBound87907.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80036.bound, LeftBound87907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound80036.actual selector witness) * (LeftBound87907.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87908

namespace LeftBound87919
def owner : Owner := ⟨.program ⟨257⟩, ⟨68195⟩⟩
def transferEvent : Nat := 87919
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 87917 .coefficient) (.value (.predecessor 1 87918 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 87917 .coefficient)
      LeftAuthority87915.bound (LeftAuthority87915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events343.exact87916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 87918 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority87915.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87915.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority87915.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound87919

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
