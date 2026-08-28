import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1759
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1787

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound265233
def owner : Owner := ⟨.program ⟨257⟩, ⟨20046⟩⟩
def transferEvent : Nat := 265233
def frameStart : Nat := 265177
def rule : BoundRule := .sum [.predecessor 0 265231 .coefficient, .predecessor 1 265232 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265231 .coefficient)
      LeftBound265216.bound (LeftBound265216.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound265216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265232 .coefficient)
      LeftAuthority265229.bound (LeftAuthority265229.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority265229.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265216.bound, LeftAuthority265229.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265216.bound, LeftAuthority265229.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265216.actual selector witness, LeftAuthority265229.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265233

namespace LeftBound265236
def owner : Owner := ⟨.program ⟨257⟩, ⟨20047⟩⟩
def transferEvent : Nat := 265236
def frameStart : Nat := 265177
def rule : BoundRule := .identity (.predecessor 0 265235 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265235 .coefficient)
      LeftBound265233.bound (LeftBound265233.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound265233.derived selector witness)

def rawBound : CoeffClass := LeftBound265233.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265233.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound265233.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound265236

namespace LeftBound265242
def owner : Owner := ⟨.program ⟨257⟩, ⟨20048⟩⟩
def transferEvent : Nat := 265242
def frameStart : Nat := 265177
def rule : BoundRule := .product (.predecessor 0 265240 .coefficient) (.predecessor 1 265241 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265240 .coefficient)
      LeftAuthority265238.bound (LeftAuthority265238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265241 .coefficient)
      LeftBound265236.bound (LeftBound265236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority265238.bound LeftBound265236.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265238.bound, LeftBound265236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority265238.actual selector witness) * (LeftBound265236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265242

namespace LeftBound265250
def owner : Owner := ⟨.program ⟨257⟩, ⟨20049⟩⟩
def transferEvent : Nat := 265250
def frameStart : Nat := 265177
def rule : BoundRule := .sum [.predecessor 0 265248 .coefficient, .predecessor 1 265249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265248 .coefficient)
      LeftAuthority265246.bound (LeftAuthority265246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265246.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265249 .coefficient)
      LeftBound265242.bound (LeftBound265242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265242.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority265246.bound, LeftBound265242.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265246.bound, LeftBound265242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority265246.actual selector witness, LeftBound265242.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265250

namespace LeftBound265254
def owner : Owner := ⟨.program ⟨257⟩, ⟨20491⟩⟩
def transferEvent : Nat := 265254
def frameStart : Nat := 265177
def rule : BoundRule := .product (.predecessor 0 265252 .coefficient) (.predecessor 1 265253 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265252 .coefficient)
      LeftBound265250.bound (LeftBound265250.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265250.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265250.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265253 .coefficient)
      LeftAuthority265227.bound (LeftAuthority265227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265227.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265227.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265250.bound LeftAuthority265227.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265250.bound, LeftAuthority265227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265250.actual selector witness) * (LeftAuthority265227.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265254

namespace LeftBound265265
def owner : Owner := ⟨.program ⟨257⟩, ⟨18769⟩⟩
def transferEvent : Nat := 265265
def frameStart : Nat := 265177
def rule : BoundRule := .product (.predecessor 0 265263 .coefficient) (.predecessor 1 265264 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265263 .coefficient)
      LeftAuthority265238.bound (LeftAuthority265238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265264 .coefficient)
      LeftAuthority265261.bound (LeftAuthority265261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265261.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority265238.bound LeftAuthority265261.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265238.bound, LeftAuthority265261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority265238.actual selector witness) * (LeftAuthority265261.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265265

namespace LeftBound265273
def owner : Owner := ⟨.program ⟨257⟩, ⟨18770⟩⟩
def transferEvent : Nat := 265273
def frameStart : Nat := 265177
def rule : BoundRule := .sum [.predecessor 0 265271 .coefficient, .predecessor 1 265272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265271 .coefficient)
      LeftAuthority265269.bound (LeftAuthority265269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265272 .coefficient)
      LeftBound265265.bound (LeftBound265265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265265.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority265269.bound, LeftBound265265.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265269.bound, LeftBound265265.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority265269.actual selector witness, LeftBound265265.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265273

namespace LeftBound265277
def owner : Owner := ⟨.program ⟨257⟩, ⟨20496⟩⟩
def transferEvent : Nat := 265277
def frameStart : Nat := 265177
def rule : BoundRule := .sum [.predecessor 0 265275 .coefficient, .predecessor 1 265276 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265275 .coefficient)
      LeftBound265273.bound (LeftBound265273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265276 .coefficient)
      LeftBound265254.bound (LeftBound265254.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265254.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265254.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265273.bound, LeftBound265254.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265273.bound, LeftBound265254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265273.actual selector witness, LeftBound265254.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265277

namespace LeftBound265290
def owner : Owner := ⟨.program ⟨257⟩, ⟨20493⟩⟩
def transferEvent : Nat := 265290
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 265288 .coefficient, .predecessor 1 265289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265288 .coefficient)
      LeftBound265119.bound (LeftBound265119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265289 .coefficient)
      LeftBound265102.bound (LeftBound265102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1035.exact265109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265102.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265119.bound, LeftBound265102.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265119.bound, LeftBound265102.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265119.actual selector witness, LeftBound265102.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265290

namespace LeftBound265293
def owner : Owner := ⟨.program ⟨257⟩, ⟨20493⟩⟩
def transferEvent : Nat := 265293
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 265287 .summary, .result 265109 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265287 .summary)
      LeftBound265121.bound (LeftBound265121.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19355⟩⟩) (rawTerms := some (Proof.Events1036.exact265287RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265109 .summary)
      LeftBound265104.bound (LeftBound265104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20492⟩⟩) (rawTerms := some (Proof.Events1035.exact265109RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265104.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound265121.bound, LeftBound265104.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265121.bound, LeftBound265104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound265121.actual selector witness, LeftBound265104.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound265293

namespace LeftBound265297
def owner : Owner := ⟨.program ⟨257⟩, ⟨20494⟩⟩
def transferEvent : Nat := 265297
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265295 .coefficient) (.predecessor 1 265296 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265295 .coefficient)
      LeftBound265290.bound (LeftBound265290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265296 .coefficient)
      LeftBound15861.bound (LeftBound15861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265290.bound LeftBound15861.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265290.bound, LeftBound15861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265290.actual selector witness) * (LeftBound15861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265297

namespace LeftBound265298
def owner : Owner := ⟨.program ⟨257⟩, ⟨20494⟩⟩
def transferEvent : Nat := 265298
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7165⟩⟩]⟩ [⟨.result 15858 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15858 .coefficient)
      LeftAuthority15857.bound (LeftAuthority15857.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7165⟩⟩) (rawTerms := some (Proof.Events061.exact15858RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15857.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15857.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15857.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265298

namespace LeftBound265299
def owner : Owner := ⟨.program ⟨257⟩, ⟨20494⟩⟩
def transferEvent : Nat := 265299
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 265294 .summary) (.transfer 265298) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265294 .summary)
      LeftBound265293.bound (LeftBound265293.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20493⟩⟩) (rawTerms := some (Proof.Events1036.exact265294RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265293.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 265298)
      LeftBound265298.bound (LeftBound265298.actual selector witness) := by
  exact .transfer (LeftBound265298.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound265293.bound LeftBound265298.bound
def bound : CoeffClass := .finite ⟨345625740372465499945107099923406305361920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265293.bound, LeftBound265298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound265293.actual selector witness) * (LeftBound265298.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265299

namespace LeftBound265314
def owner : Owner := ⟨.program ⟨257⟩, ⟨17616⟩⟩
def transferEvent : Nat := 265314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 265312 .coefficient) (.predecessor 1 265313 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 265312 .coefficient)
      LeftBound259871.bound (LeftBound259871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1015.exact259875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound259871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound259871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 265313 .coefficient)
      LeftAuthority265310.bound (LeftAuthority265310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1036.exact265311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259871.bound LeftAuthority265310.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259871.bound, LeftAuthority265310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259871.actual selector witness) * (LeftAuthority265310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265314

namespace LeftBound265315
def owner : Owner := ⟨.program ⟨257⟩, ⟨17616⟩⟩
def transferEvent : Nat := 265315
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨17614⟩⟩]⟩ [⟨.result 265311 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265311 .coefficient)
      LeftAuthority265310.bound (LeftAuthority265310.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨17614⟩⟩) (rawTerms := some (Proof.Events1036.exact265311RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority265310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority265310.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority265310.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority265310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority265310.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound265315

namespace LeftBound265316
def owner : Owner := ⟨.program ⟨257⟩, ⟨17616⟩⟩
def transferEvent : Nat := 265316
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 259875 .summary) (.transfer 265315) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 259875 .summary)
      LeftBound259874.bound (LeftBound259874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17306⟩⟩) (rawTerms := some (Proof.Events1015.exact259875RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound259874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 265315)
      LeftBound265315.bound (LeftBound265315.actual selector witness) := by
  exact .transfer (LeftBound265315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound259874.bound LeftBound265315.bound
def bound : CoeffClass := .finite ⟨32188807212483504816668771614720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound259874.bound, LeftBound265315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound259874.actual selector witness) * (LeftBound265315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound265316

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
