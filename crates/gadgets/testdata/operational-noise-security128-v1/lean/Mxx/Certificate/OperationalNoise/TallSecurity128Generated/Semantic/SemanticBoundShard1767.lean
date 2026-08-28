import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1701
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1766

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound261841
def owner : Owner := ⟨.program ⟨257⟩, ⟨49486⟩⟩
def transferEvent : Nat := 261841
def frameStart : Nat := 261785
def rule : BoundRule := .sum [.predecessor 0 261839 .coefficient, .predecessor 1 261840 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261839 .coefficient)
      LeftBound261824.bound (LeftBound261824.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound261824.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261840 .coefficient)
      LeftAuthority261837.bound (LeftAuthority261837.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority261837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261824.bound, LeftAuthority261837.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261824.bound, LeftAuthority261837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261824.actual selector witness, LeftAuthority261837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261841

namespace LeftBound261844
def owner : Owner := ⟨.program ⟨257⟩, ⟨49487⟩⟩
def transferEvent : Nat := 261844
def frameStart : Nat := 261785
def rule : BoundRule := .identity (.predecessor 0 261843 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261843 .coefficient)
      LeftBound261841.bound (LeftBound261841.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound261841.derived selector witness)

def rawBound : CoeffClass := LeftBound261841.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound261841.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound261844

namespace LeftBound261850
def owner : Owner := ⟨.program ⟨257⟩, ⟨49488⟩⟩
def transferEvent : Nat := 261850
def frameStart : Nat := 261785
def rule : BoundRule := .product (.predecessor 0 261848 .coefficient) (.predecessor 1 261849 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261848 .coefficient)
      LeftAuthority261846.bound (LeftAuthority261846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261849 .coefficient)
      LeftBound261844.bound (LeftBound261844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority261846.bound LeftBound261844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261846.bound, LeftBound261844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority261846.actual selector witness) * (LeftBound261844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261850

namespace LeftBound261858
def owner : Owner := ⟨.program ⟨257⟩, ⟨49489⟩⟩
def transferEvent : Nat := 261858
def frameStart : Nat := 261785
def rule : BoundRule := .sum [.predecessor 0 261856 .coefficient, .predecessor 1 261857 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261856 .coefficient)
      LeftAuthority261854.bound (LeftAuthority261854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261857 .coefficient)
      LeftBound261850.bound (LeftBound261850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261850.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority261854.bound, LeftBound261850.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261854.bound, LeftBound261850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority261854.actual selector witness, LeftBound261850.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261858

namespace LeftBound261862
def owner : Owner := ⟨.program ⟨257⟩, ⟨49899⟩⟩
def transferEvent : Nat := 261862
def frameStart : Nat := 261785
def rule : BoundRule := .product (.predecessor 0 261860 .coefficient) (.predecessor 1 261861 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261860 .coefficient)
      LeftBound261858.bound (LeftBound261858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261861 .coefficient)
      LeftAuthority261835.bound (LeftAuthority261835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261835.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound261858.bound LeftAuthority261835.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261858.bound, LeftAuthority261835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound261858.actual selector witness) * (LeftAuthority261835.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261862

namespace LeftBound261873
def owner : Owner := ⟨.program ⟨257⟩, ⟨48296⟩⟩
def transferEvent : Nat := 261873
def frameStart : Nat := 261785
def rule : BoundRule := .product (.predecessor 0 261871 .coefficient) (.predecessor 1 261872 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261871 .coefficient)
      LeftAuthority261846.bound (LeftAuthority261846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261846.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261872 .coefficient)
      LeftAuthority261869.bound (LeftAuthority261869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority261846.bound LeftAuthority261869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261846.bound, LeftAuthority261869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority261846.actual selector witness) * (LeftAuthority261869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261873

namespace LeftBound261881
def owner : Owner := ⟨.program ⟨257⟩, ⟨48297⟩⟩
def transferEvent : Nat := 261881
def frameStart : Nat := 261785
def rule : BoundRule := .sum [.predecessor 0 261879 .coefficient, .predecessor 1 261880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261879 .coefficient)
      LeftAuthority261877.bound (LeftAuthority261877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261880 .coefficient)
      LeftBound261873.bound (LeftBound261873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority261877.bound, LeftBound261873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261877.bound, LeftBound261873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority261877.actual selector witness, LeftBound261873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261881

namespace LeftBound261885
def owner : Owner := ⟨.program ⟨257⟩, ⟨49903⟩⟩
def transferEvent : Nat := 261885
def frameStart : Nat := 261785
def rule : BoundRule := .sum [.predecessor 0 261883 .coefficient, .predecessor 1 261884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261883 .coefficient)
      LeftBound261881.bound (LeftBound261881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261884 .coefficient)
      LeftBound261862.bound (LeftBound261862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261867RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261881.bound, LeftBound261862.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261881.bound, LeftBound261862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261881.actual selector witness, LeftBound261862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261885

namespace LeftBound261898
def owner : Owner := ⟨.program ⟨257⟩, ⟨49901⟩⟩
def transferEvent : Nat := 261898
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 261896 .coefficient, .predecessor 1 261897 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261896 .coefficient)
      LeftBound261727.bound (LeftBound261727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1023.exact261895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261897 .coefficient)
      LeftBound261710.bound (LeftBound261710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1022.exact261717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261727.bound, LeftBound261710.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261727.bound, LeftBound261710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261727.actual selector witness, LeftBound261710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261898

namespace LeftBound261901
def owner : Owner := ⟨.program ⟨257⟩, ⟨49901⟩⟩
def transferEvent : Nat := 261901
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 261895 .summary, .result 261717 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261895 .summary)
      LeftBound261729.bound (LeftBound261729.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48795⟩⟩) (rawTerms := some (Proof.Events1023.exact261895RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound261729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261717 .summary)
      LeftBound261712.bound (LeftBound261712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49900⟩⟩) (rawTerms := some (Proof.Events1022.exact261717RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound261712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound261729.bound, LeftBound261712.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261729.bound, LeftBound261712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound261729.actual selector witness, LeftBound261712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound261901

namespace LeftBound261905
def owner : Owner := ⟨.program ⟨257⟩, ⟨49902⟩⟩
def transferEvent : Nat := 261905
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 261903 .coefficient) (.predecessor 1 261904 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261903 .coefficient)
      LeftBound261898.bound (LeftBound261898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1023.exact261902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound261898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound261898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261904 .coefficient)
      LeftBound15541.bound (LeftBound15541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound261898.bound LeftBound15541.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261898.bound, LeftBound15541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound261898.actual selector witness) * (LeftBound15541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261905

namespace LeftBound261906
def owner : Owner := ⟨.program ⟨257⟩, ⟨49902⟩⟩
def transferEvent : Nat := 261906
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩ [⟨.result 15538 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15538 .coefficient)
      LeftAuthority15537.bound (LeftAuthority15537.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7147⟩⟩) (rawTerms := some (Proof.Events060.exact15538RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15537.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15537.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15537.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound261906

namespace LeftBound261907
def owner : Owner := ⟨.program ⟨257⟩, ⟨49902⟩⟩
def transferEvent : Nat := 261907
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 261902 .summary) (.transfer 261906) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261902 .summary)
      LeftBound261901.bound (LeftBound261901.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49901⟩⟩) (rawTerms := some (Proof.Events1023.exact261902RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound261901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 261906)
      LeftBound261906.bound (LeftBound261906.actual selector witness) := by
  exact .transfer (LeftBound261906.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound261901.bound LeftBound261906.bound
def bound : CoeffClass := .finite ⟨345685857434530723496243679576218056785920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound261901.bound, LeftBound261906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound261901.actual selector witness) * (LeftBound261906.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261907

namespace LeftBound261922
def owner : Owner := ⟨.program ⟨257⟩, ⟨47220⟩⟩
def transferEvent : Nat := 261922
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 261920 .coefficient) (.predecessor 1 261921 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 261920 .coefficient)
      LeftBound252159.bound (LeftBound252159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events985.exact252163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound252159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound252159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 261921 .coefficient)
      LeftAuthority261918.bound (LeftAuthority261918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1023.exact261919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261918.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound252159.bound LeftAuthority261918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252159.bound, LeftAuthority261918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound252159.actual selector witness) * (LeftAuthority261918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261922

namespace LeftBound261923
def owner : Owner := ⟨.program ⟨257⟩, ⟨47220⟩⟩
def transferEvent : Nat := 261923
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47218⟩⟩]⟩ [⟨.result 261919 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 261919 .coefficient)
      LeftAuthority261918.bound (LeftAuthority261918.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47218⟩⟩) (rawTerms := some (Proof.Events1023.exact261919RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority261918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority261918.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority261918.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority261918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority261918.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound261923

namespace LeftBound261924
def owner : Owner := ⟨.program ⟨257⟩, ⟨47220⟩⟩
def transferEvent : Nat := 261924
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 252163 .summary) (.transfer 261923) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 252163 .summary)
      LeftBound252162.bound (LeftBound252162.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46926⟩⟩) (rawTerms := some (Proof.Events985.exact252163RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound252162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 261923)
      LeftBound261923.bound (LeftBound261923.actual selector witness) := by
  exact .transfer (LeftBound261923.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound252162.bound LeftBound261923.bound
def bound : CoeffClass := .finite ⟨32194307824962751379413684715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound252162.bound, LeftBound261923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound252162.actual selector witness) * (LeftBound261923.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound261924

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
