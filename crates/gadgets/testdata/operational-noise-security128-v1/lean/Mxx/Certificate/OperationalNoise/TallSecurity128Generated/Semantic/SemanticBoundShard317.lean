import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard316

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52131
def owner : Owner := ⟨.program ⟨257⟩, ⟨56722⟩⟩
def transferEvent : Nat := 52131
def frameStart : Nat := 52102
def rule : BoundRule := .product (.predecessor 0 52129 .coefficient) (.predecessor 1 52130 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52129 .coefficient)
      LeftAuthority52127.bound (LeftAuthority52127.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52127.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52127.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52130 .coefficient)
      LeftAuthority52124.bound (LeftAuthority52124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52124.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52127.bound LeftAuthority52124.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52127.bound, LeftAuthority52124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority52127.actual selector witness) * (LeftAuthority52124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52131

namespace LeftBound52135
def owner : Owner := ⟨.program ⟨257⟩, ⟨56723⟩⟩
def transferEvent : Nat := 52135
def frameStart : Nat := 52102
def rule : BoundRule := .identity (.predecessor 0 52134 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52134 .coefficient)
      LeftBound52131.bound (LeftBound52131.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52131.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52131.derived selector witness)

def rawBound : CoeffClass := LeftBound52131.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52131.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound52131.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52135

namespace LeftBound52152
def owner : Owner := ⟨.program ⟨257⟩, ⟨58278⟩⟩
def transferEvent : Nat := 52152
def frameStart : Nat := 52102
def rule : BoundRule := .sum [.predecessor 0 52150 .coefficient, .predecessor 1 52151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52150 .coefficient)
      LeftBound52135.bound (LeftBound52135.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52151 .coefficient)
      LeftAuthority52148.bound (LeftAuthority52148.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52135.bound, LeftAuthority52148.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52135.bound, LeftAuthority52148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound52135.actual selector witness, LeftAuthority52148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52152

namespace LeftBound52155
def owner : Owner := ⟨.program ⟨257⟩, ⟨58279⟩⟩
def transferEvent : Nat := 52155
def frameStart : Nat := 52102
def rule : BoundRule := .identity (.predecessor 0 52154 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52154 .coefficient)
      LeftBound52152.bound (LeftBound52152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound52152.derived selector witness)

def rawBound : CoeffClass := LeftBound52152.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52152.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound52152.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52155

namespace LeftBound52161
def owner : Owner := ⟨.program ⟨257⟩, ⟨58280⟩⟩
def transferEvent : Nat := 52161
def frameStart : Nat := 52102
def rule : BoundRule := .product (.predecessor 0 52159 .coefficient) (.predecessor 1 52160 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52159 .coefficient)
      LeftAuthority52157.bound (LeftAuthority52157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52160 .coefficient)
      LeftBound52155.bound (LeftBound52155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority52157.bound LeftBound52155.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52157.bound, LeftBound52155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority52157.actual selector witness) * (LeftBound52155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52161

namespace LeftBound52177
def owner : Owner := ⟨.program ⟨257⟩, ⟨9533⟩⟩
def transferEvent : Nat := 52177
def frameStart : Nat := 52102
def rule : BoundRule := .scale (.predecessor 0 52175 .coefficient) (.value (.predecessor 1 52176 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52175 .coefficient)
      LeftAuthority52173.bound (LeftAuthority52173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52176 .coefficient)
      LeftAuthority52164.bound (LeftAuthority52164.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority52164.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority52173.bound LeftAuthority52164.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52173.bound, LeftAuthority52164.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority52173.actual selector witness) * (LeftAuthority52164.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52177

namespace LeftBound52180
def owner : Owner := ⟨.program ⟨257⟩, ⟨7290⟩⟩
def transferEvent : Nat := 52180
def frameStart : Nat := 52102
def rule : BoundRule := .identity (.predecessor 0 52179 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52179 .coefficient)
      LeftAuthority52167.bound (LeftAuthority52167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52167.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52167.derived selector witness)

def rawBound : CoeffClass := LeftAuthority52167.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority52167.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound52180

namespace LeftBound52184
def owner : Owner := ⟨.program ⟨257⟩, ⟨9534⟩⟩
def transferEvent : Nat := 52184
def frameStart : Nat := 52102
def rule : BoundRule := .product (.predecessor 0 52182 .coefficient) (.predecessor 1 52183 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52182 .coefficient)
      LeftBound52180.bound (LeftBound52180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52183 .coefficient)
      LeftBound52177.bound (LeftBound52177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound52180.bound LeftBound52177.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52180.bound, LeftBound52177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound52180.actual selector witness) * (LeftBound52177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52184

namespace LeftBound52189
def owner : Owner := ⟨.program ⟨257⟩, ⟨58281⟩⟩
def transferEvent : Nat := 52189
def frameStart : Nat := 52102
def rule : BoundRule := .sum [.predecessor 0 52187 .coefficient, .predecessor 1 52188 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52187 .coefficient)
      LeftBound52184.bound (LeftBound52184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52188 .coefficient)
      LeftBound52161.bound (LeftBound52161.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52161.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52161.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52184.bound, LeftBound52161.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52184.bound, LeftBound52161.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound52184.actual selector witness, LeftBound52161.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52189

namespace LeftBound52193
def owner : Owner := ⟨.program ⟨257⟩, ⟨58570⟩⟩
def transferEvent : Nat := 52193
def frameStart : Nat := 52102
def rule : BoundRule := .product (.predecessor 0 52191 .coefficient) (.predecessor 1 52192 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52191 .coefficient)
      LeftBound52189.bound (LeftBound52189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52189.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52192 .coefficient)
      LeftAuthority52146.bound (LeftAuthority52146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound52189.bound LeftAuthority52146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52189.bound, LeftAuthority52146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound52189.actual selector witness) * (LeftAuthority52146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52193

namespace LeftBound52204
def owner : Owner := ⟨.program ⟨257⟩, ⟨56914⟩⟩
def transferEvent : Nat := 52204
def frameStart : Nat := 52102
def rule : BoundRule := .product (.predecessor 0 52202 .coefficient) (.predecessor 1 52203 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52202 .coefficient)
      LeftAuthority52157.bound (LeftAuthority52157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52158RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52157.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52203 .coefficient)
      LeftAuthority52200.bound (LeftAuthority52200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52157.bound LeftAuthority52200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52157.bound, LeftAuthority52200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority52157.actual selector witness) * (LeftAuthority52200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52204

namespace LeftBound52212
def owner : Owner := ⟨.program ⟨257⟩, ⟨56915⟩⟩
def transferEvent : Nat := 52212
def frameStart : Nat := 52102
def rule : BoundRule := .sum [.predecessor 0 52210 .coefficient, .predecessor 1 52211 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52210 .coefficient)
      LeftAuthority52208.bound (LeftAuthority52208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52211 .coefficient)
      LeftBound52204.bound (LeftBound52204.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52206RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52204.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52204.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52208.bound, LeftBound52204.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52208.bound, LeftBound52204.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority52208.actual selector witness, LeftBound52204.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52212

namespace LeftBound52216
def owner : Owner := ⟨.program ⟨257⟩, ⟨58571⟩⟩
def transferEvent : Nat := 52216
def frameStart : Nat := 52102
def rule : BoundRule := .sum [.predecessor 0 52214 .coefficient, .predecessor 1 52215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52214 .coefficient)
      LeftBound52212.bound (LeftBound52212.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52212.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52215 .coefficient)
      LeftBound52193.bound (LeftBound52193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52212.bound, LeftBound52193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52212.bound, LeftBound52193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound52212.actual selector witness, LeftBound52193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52216

namespace LeftBound52229
def owner : Owner := ⟨.program ⟨257⟩, ⟨58569⟩⟩
def transferEvent : Nat := 52229
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52227 .coefficient, .predecessor 1 52228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52227 .coefficient)
      LeftBound52050.bound (LeftBound52050.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52050.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52228 .coefficient)
      LeftBound52033.bound (LeftBound52033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events203.exact52040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52033.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52050.bound, LeftBound52033.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52050.bound, LeftBound52033.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound52050.actual selector witness, LeftBound52033.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52229

namespace LeftBound52232
def owner : Owner := ⟨.program ⟨257⟩, ⟨58569⟩⟩
def transferEvent : Nat := 52232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 52226 .summary, .result 52040 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 52226 .summary)
      LeftBound52052.bound (LeftBound52052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨57492⟩⟩) (rawTerms := some (Proof.Events204.exact52226RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 52040 .summary)
      LeftBound52035.bound (LeftBound52035.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58568⟩⟩) (rawTerms := some (Proof.Events203.exact52040RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52052.bound, LeftBound52035.bound]
def bound : CoeffClass := .finite ⟨2997944351807545540608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52052.bound, LeftBound52035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound52052.actual selector witness, LeftBound52035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52232

namespace LeftBound52236
def owner : Owner := ⟨.program ⟨257⟩, ⟨59162⟩⟩
def transferEvent : Nat := 52236
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52234 .coefficient) (.predecessor 1 52235 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 52234 .coefficient)
      LeftBound52229.bound (LeftBound52229.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52229.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 52235 .coefficient)
      LeftAuthority51955.bound (LeftAuthority51955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events202.exact51956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51955.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51955.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound52229.bound LeftAuthority51955.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52229.bound, LeftAuthority51955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound52229.actual selector witness) * (LeftAuthority51955.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52236

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
