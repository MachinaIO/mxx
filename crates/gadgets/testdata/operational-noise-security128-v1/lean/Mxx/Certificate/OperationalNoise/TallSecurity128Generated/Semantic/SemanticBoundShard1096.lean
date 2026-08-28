import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1095

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound164814
def owner : Owner := ⟨.program ⟨257⟩, ⟨44082⟩⟩
def transferEvent : Nat := 164814
def frameStart : Nat := 164764
def rule : BoundRule := .sum [.predecessor 0 164812 .coefficient, .predecessor 1 164813 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164812 .coefficient)
      LeftBound164797.bound (LeftBound164797.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound164797.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164813 .coefficient)
      LeftAuthority164810.bound (LeftAuthority164810.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority164810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164797.bound, LeftAuthority164810.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164797.bound, LeftAuthority164810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164797.actual selector witness, LeftAuthority164810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164814

namespace LeftBound164817
def owner : Owner := ⟨.program ⟨257⟩, ⟨44083⟩⟩
def transferEvent : Nat := 164817
def frameStart : Nat := 164764
def rule : BoundRule := .identity (.predecessor 0 164816 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164816 .coefficient)
      LeftBound164814.bound (LeftBound164814.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound164814.derived selector witness)

def rawBound : CoeffClass := LeftBound164814.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound164814.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound164817

namespace LeftBound164823
def owner : Owner := ⟨.program ⟨257⟩, ⟨44084⟩⟩
def transferEvent : Nat := 164823
def frameStart : Nat := 164764
def rule : BoundRule := .product (.predecessor 0 164821 .coefficient) (.predecessor 1 164822 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164821 .coefficient)
      LeftAuthority164819.bound (LeftAuthority164819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164822 .coefficient)
      LeftBound164817.bound (LeftBound164817.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164818RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164817.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164817.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority164819.bound LeftBound164817.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164819.bound, LeftBound164817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority164819.actual selector witness) * (LeftBound164817.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164823

namespace LeftBound164839
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 164839
def frameStart : Nat := 164764
def rule : BoundRule := .scale (.predecessor 0 164837 .coefficient) (.value (.predecessor 1 164838 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164837 .coefficient)
      LeftAuthority164835.bound (LeftAuthority164835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164838 .coefficient)
      LeftAuthority164826.bound (LeftAuthority164826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority164826.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority164835.bound LeftAuthority164826.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164835.bound, LeftAuthority164826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164835.actual selector witness) * (LeftAuthority164826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound164839

namespace LeftBound164842
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 164842
def frameStart : Nat := 164764
def rule : BoundRule := .identity (.predecessor 0 164841 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164841 .coefficient)
      LeftAuthority164829.bound (LeftAuthority164829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164829.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164829.derived selector witness)

def rawBound : CoeffClass := LeftAuthority164829.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority164829.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound164842

namespace LeftBound164846
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 164846
def frameStart : Nat := 164764
def rule : BoundRule := .product (.predecessor 0 164844 .coefficient) (.predecessor 1 164845 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164844 .coefficient)
      LeftBound164842.bound (LeftBound164842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164845 .coefficient)
      LeftBound164839.bound (LeftBound164839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164842.bound LeftBound164839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164842.bound, LeftBound164839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164842.actual selector witness) * (LeftBound164839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164846

namespace LeftBound164851
def owner : Owner := ⟨.program ⟨257⟩, ⟨44085⟩⟩
def transferEvent : Nat := 164851
def frameStart : Nat := 164764
def rule : BoundRule := .sum [.predecessor 0 164849 .coefficient, .predecessor 1 164850 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164849 .coefficient)
      LeftBound164846.bound (LeftBound164846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164850 .coefficient)
      LeftBound164823.bound (LeftBound164823.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164825RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164823.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164823.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164846.bound, LeftBound164823.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164846.bound, LeftBound164823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164846.actual selector witness, LeftBound164823.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164851

namespace LeftBound164855
def owner : Owner := ⟨.program ⟨257⟩, ⟨44346⟩⟩
def transferEvent : Nat := 164855
def frameStart : Nat := 164764
def rule : BoundRule := .product (.predecessor 0 164853 .coefficient) (.predecessor 1 164854 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164853 .coefficient)
      LeftBound164851.bound (LeftBound164851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164851.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164854 .coefficient)
      LeftAuthority164808.bound (LeftAuthority164808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164851.bound LeftAuthority164808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164851.bound, LeftAuthority164808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164851.actual selector witness) * (LeftAuthority164808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164855

namespace LeftBound164866
def owner : Owner := ⟨.program ⟨257⟩, ⟨42822⟩⟩
def transferEvent : Nat := 164866
def frameStart : Nat := 164764
def rule : BoundRule := .product (.predecessor 0 164864 .coefficient) (.predecessor 1 164865 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164864 .coefficient)
      LeftAuthority164819.bound (LeftAuthority164819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164819.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164819.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164865 .coefficient)
      LeftAuthority164862.bound (LeftAuthority164862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164862.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority164819.bound LeftAuthority164862.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164819.bound, LeftAuthority164862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority164819.actual selector witness) * (LeftAuthority164862.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164866

namespace LeftBound164874
def owner : Owner := ⟨.program ⟨257⟩, ⟨42823⟩⟩
def transferEvent : Nat := 164874
def frameStart : Nat := 164764
def rule : BoundRule := .sum [.predecessor 0 164872 .coefficient, .predecessor 1 164873 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164872 .coefficient)
      LeftAuthority164870.bound (LeftAuthority164870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events644.exact164871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164870.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164873 .coefficient)
      LeftBound164866.bound (LeftBound164866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events644.exact164868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority164870.bound, LeftBound164866.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164870.bound, LeftBound164866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority164870.actual selector witness, LeftBound164866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164874

namespace LeftBound164878
def owner : Owner := ⟨.program ⟨257⟩, ⟨44347⟩⟩
def transferEvent : Nat := 164878
def frameStart : Nat := 164764
def rule : BoundRule := .sum [.predecessor 0 164876 .coefficient, .predecessor 1 164877 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164876 .coefficient)
      LeftBound164874.bound (LeftBound164874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events644.exact164875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164877 .coefficient)
      LeftBound164855.bound (LeftBound164855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164874.bound, LeftBound164855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164874.bound, LeftBound164855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164874.actual selector witness, LeftBound164855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164878

namespace LeftBound164891
def owner : Owner := ⟨.program ⟨257⟩, ⟨44345⟩⟩
def transferEvent : Nat := 164891
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 164889 .coefficient, .predecessor 1 164890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164889 .coefficient)
      LeftBound164712.bound (LeftBound164712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events644.exact164888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164890 .coefficient)
      LeftBound164695.bound (LeftBound164695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164712.bound, LeftBound164695.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164712.bound, LeftBound164695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164712.actual selector witness, LeftBound164695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164891

namespace LeftBound164894
def owner : Owner := ⟨.program ⟨257⟩, ⟨44345⟩⟩
def transferEvent : Nat := 164894
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 164888 .summary, .result 164702 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164888 .summary)
      LeftBound164714.bound (LeftBound164714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43272⟩⟩) (rawTerms := some (Proof.Events644.exact164888RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164702 .summary)
      LeftBound164697.bound (LeftBound164697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44344⟩⟩) (rawTerms := some (Proof.Events643.exact164702RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164714.bound, LeftBound164697.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164714.bound, LeftBound164697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164714.actual selector witness, LeftBound164697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164894

namespace LeftBound164898
def owner : Owner := ⟨.program ⟨257⟩, ⟨44771⟩⟩
def transferEvent : Nat := 164898
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164896 .coefficient) (.predecessor 1 164897 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164896 .coefficient)
      LeftBound164891.bound (LeftBound164891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events644.exact164895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164897 .coefficient)
      LeftAuthority164617.bound (LeftAuthority164617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events643.exact164618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164617.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164891.bound LeftAuthority164617.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164891.bound, LeftAuthority164617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164891.actual selector witness) * (LeftAuthority164617.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164898

namespace LeftBound164899
def owner : Owner := ⟨.program ⟨257⟩, ⟨44771⟩⟩
def transferEvent : Nat := 164899
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨44769⟩⟩]⟩ [⟨.result 164618 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164618 .coefficient)
      LeftAuthority164617.bound (LeftAuthority164617.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨44769⟩⟩) (rawTerms := some (Proof.Events643.exact164618RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164617.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority164617.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164617.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164899

namespace LeftBound164900
def owner : Owner := ⟨.program ⟨257⟩, ⟨44771⟩⟩
def transferEvent : Nat := 164900
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 164895 .summary) (.transfer 164899) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164895 .summary)
      LeftBound164894.bound (LeftBound164894.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44345⟩⟩) (rawTerms := some (Proof.Events644.exact164895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 164899)
      LeftBound164899.bound (LeftBound164899.actual selector witness) := by
  exact .transfer (LeftBound164899.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164894.bound LeftBound164899.bound
def bound : CoeffClass := .finite ⟨32193718473625689247691015454720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164894.bound, LeftBound164899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164894.actual selector witness) * (LeftBound164899.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164900

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
