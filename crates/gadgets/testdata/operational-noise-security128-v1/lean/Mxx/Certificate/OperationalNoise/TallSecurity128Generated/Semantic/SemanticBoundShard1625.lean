import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1624

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound240834
def owner : Owner := ⟨.program ⟨257⟩, ⟨68920⟩⟩
def transferEvent : Nat := 240834
def frameStart : Nat := 240781
def rule : BoundRule := .identity (.predecessor 0 240833 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240833 .coefficient)
      LeftBound240831.bound (LeftBound240831.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound240831.derived selector witness)

def rawBound : CoeffClass := LeftBound240831.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound240831.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound240834

namespace LeftBound240840
def owner : Owner := ⟨.program ⟨257⟩, ⟨68921⟩⟩
def transferEvent : Nat := 240840
def frameStart : Nat := 240781
def rule : BoundRule := .product (.predecessor 0 240838 .coefficient) (.predecessor 1 240839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240838 .coefficient)
      LeftAuthority240836.bound (LeftAuthority240836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240839 .coefficient)
      LeftBound240834.bound (LeftBound240834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority240836.bound LeftBound240834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240836.bound, LeftBound240834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority240836.actual selector witness) * (LeftBound240834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240840

namespace LeftBound240856
def owner : Owner := ⟨.program ⟨257⟩, ⟨9542⟩⟩
def transferEvent : Nat := 240856
def frameStart : Nat := 240781
def rule : BoundRule := .scale (.predecessor 0 240854 .coefficient) (.value (.predecessor 1 240855 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240854 .coefficient)
      LeftAuthority240852.bound (LeftAuthority240852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240853RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240852.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240855 .coefficient)
      LeftAuthority240843.bound (LeftAuthority240843.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority240843.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority240852.bound LeftAuthority240843.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240852.bound, LeftAuthority240843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority240852.actual selector witness) * (LeftAuthority240843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound240856

namespace LeftBound240859
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 240859
def frameStart : Nat := 240781
def rule : BoundRule := .identity (.predecessor 0 240858 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240858 .coefficient)
      LeftAuthority240846.bound (LeftAuthority240846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240846.derived selector witness)

def rawBound : CoeffClass := LeftAuthority240846.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority240846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound240859

namespace LeftBound240863
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 240863
def frameStart : Nat := 240781
def rule : BoundRule := .product (.predecessor 0 240861 .coefficient) (.predecessor 1 240862 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240861 .coefficient)
      LeftBound240859.bound (LeftBound240859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240862 .coefficient)
      LeftBound240856.bound (LeftBound240856.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240856.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240856.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound240859.bound LeftBound240856.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240859.bound, LeftBound240856.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound240859.actual selector witness) * (LeftBound240856.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240863

namespace LeftBound240868
def owner : Owner := ⟨.program ⟨257⟩, ⟨68922⟩⟩
def transferEvent : Nat := 240868
def frameStart : Nat := 240781
def rule : BoundRule := .sum [.predecessor 0 240866 .coefficient, .predecessor 1 240867 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240866 .coefficient)
      LeftBound240863.bound (LeftBound240863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240867 .coefficient)
      LeftBound240840.bound (LeftBound240840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound240863.bound, LeftBound240840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240863.bound, LeftBound240840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound240863.actual selector witness, LeftBound240840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound240868

namespace LeftBound240872
def owner : Owner := ⟨.program ⟨257⟩, ⟨69221⟩⟩
def transferEvent : Nat := 240872
def frameStart : Nat := 240781
def rule : BoundRule := .product (.predecessor 0 240870 .coefficient) (.predecessor 1 240871 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240870 .coefficient)
      LeftBound240868.bound (LeftBound240868.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240869RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240868.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240871 .coefficient)
      LeftAuthority240825.bound (LeftAuthority240825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound240868.bound LeftAuthority240825.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240868.bound, LeftAuthority240825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound240868.actual selector witness) * (LeftAuthority240825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240872

namespace LeftBound240883
def owner : Owner := ⟨.program ⟨257⟩, ⟨65774⟩⟩
def transferEvent : Nat := 240883
def frameStart : Nat := 240781
def rule : BoundRule := .product (.predecessor 0 240881 .coefficient) (.predecessor 1 240882 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240881 .coefficient)
      LeftAuthority240836.bound (LeftAuthority240836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240882 .coefficient)
      LeftAuthority240879.bound (LeftAuthority240879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority240836.bound LeftAuthority240879.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240836.bound, LeftAuthority240879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority240836.actual selector witness) * (LeftAuthority240879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240883

namespace LeftBound240891
def owner : Owner := ⟨.program ⟨257⟩, ⟨65775⟩⟩
def transferEvent : Nat := 240891
def frameStart : Nat := 240781
def rule : BoundRule := .sum [.predecessor 0 240889 .coefficient, .predecessor 1 240890 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240889 .coefficient)
      LeftAuthority240887.bound (LeftAuthority240887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240887.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240890 .coefficient)
      LeftBound240883.bound (LeftBound240883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority240887.bound, LeftBound240883.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240887.bound, LeftBound240883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority240887.actual selector witness, LeftBound240883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound240891

namespace LeftBound240895
def owner : Owner := ⟨.program ⟨257⟩, ⟨69222⟩⟩
def transferEvent : Nat := 240895
def frameStart : Nat := 240781
def rule : BoundRule := .sum [.predecessor 0 240893 .coefficient, .predecessor 1 240894 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240893 .coefficient)
      LeftBound240891.bound (LeftBound240891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240891.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240894 .coefficient)
      LeftBound240872.bound (LeftBound240872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240872.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound240891.bound, LeftBound240872.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240891.bound, LeftBound240872.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound240891.actual selector witness, LeftBound240872.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound240895

namespace LeftBound240908
def owner : Owner := ⟨.program ⟨257⟩, ⟨69220⟩⟩
def transferEvent : Nat := 240908
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 240906 .coefficient, .predecessor 1 240907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240906 .coefficient)
      LeftBound240729.bound (LeftBound240729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events941.exact240905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240907 .coefficient)
      LeftBound240712.bound (LeftBound240712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events940.exact240719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240712.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound240729.bound, LeftBound240712.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240729.bound, LeftBound240712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound240729.actual selector witness, LeftBound240712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound240908

namespace LeftBound240911
def owner : Owner := ⟨.program ⟨257⟩, ⟨69220⟩⟩
def transferEvent : Nat := 240911
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 240905 .summary, .result 240719 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240905 .summary)
      LeftBound240731.bound (LeftBound240731.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67753⟩⟩) (rawTerms := some (Proof.Events941.exact240905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound240731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240719 .summary)
      LeftBound240714.bound (LeftBound240714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69219⟩⟩) (rawTerms := some (Proof.Events940.exact240719RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound240714.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound240731.bound, LeftBound240714.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240731.bound, LeftBound240714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound240731.actual selector witness, LeftBound240714.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound240911

namespace LeftBound240915
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def transferEvent : Nat := 240915
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 240913 .coefficient) (.predecessor 1 240914 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240913 .coefficient)
      LeftBound240908.bound (LeftBound240908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events941.exact240912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound240908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound240908.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240914 .coefficient)
      LeftAuthority240634.bound (LeftAuthority240634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events939.exact240635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240634.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound240908.bound LeftAuthority240634.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240908.bound, LeftAuthority240634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound240908.actual selector witness) * (LeftAuthority240634.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240915

namespace LeftBound240916
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def transferEvent : Nat := 240916
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70019⟩⟩]⟩ [⟨.result 240635 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240635 .coefficient)
      LeftAuthority240634.bound (LeftAuthority240634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70019⟩⟩) (rawTerms := some (Proof.Events939.exact240635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority240634.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority240634.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound240916

namespace LeftBound240917
def owner : Owner := ⟨.program ⟨257⟩, ⟨70021⟩⟩
def transferEvent : Nat := 240917
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 240912 .summary) (.transfer 240916) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 240912 .summary)
      LeftBound240911.bound (LeftBound240911.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69220⟩⟩) (rawTerms := some (Proof.Events941.exact240912RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound240911.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 240916)
      LeftBound240916.bound (LeftBound240916.actual selector witness) := by
  exact .transfer (LeftBound240916.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound240911.bound LeftBound240916.bound
def bound : CoeffClass := .finite ⟨32191361068277440720800338411520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound240911.bound, LeftBound240916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound240911.actual selector witness) * (LeftBound240916.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound240917

namespace LeftBound240928
def owner : Owner := ⟨.program ⟨257⟩, ⟨68039⟩⟩
def transferEvent : Nat := 240928
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 240926 .coefficient) (.value (.predecessor 1 240927 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 240926 .coefficient)
      LeftAuthority240924.bound (LeftAuthority240924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events941.exact240925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority240924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority240924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 240927 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority240924.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority240924.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority240924.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound240928

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
