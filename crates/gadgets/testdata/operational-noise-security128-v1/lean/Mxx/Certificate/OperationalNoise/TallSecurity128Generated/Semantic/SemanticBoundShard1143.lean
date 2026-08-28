import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1142

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound171059
def owner : Owner := ⟨.program ⟨257⟩, ⟨21591⟩⟩
def transferEvent : Nat := 171059
def frameStart : Nat := 171030
def rule : BoundRule := .product (.predecessor 0 171057 .coefficient) (.predecessor 1 171058 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171057 .coefficient)
      LeftAuthority171055.bound (LeftAuthority171055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171055.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171058 .coefficient)
      LeftAuthority171052.bound (LeftAuthority171052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority171055.bound LeftAuthority171052.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171055.bound, LeftAuthority171052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority171055.actual selector witness) * (LeftAuthority171052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171059

namespace LeftBound171063
def owner : Owner := ⟨.program ⟨257⟩, ⟨21592⟩⟩
def transferEvent : Nat := 171063
def frameStart : Nat := 171030
def rule : BoundRule := .identity (.predecessor 0 171062 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171062 .coefficient)
      LeftBound171059.bound (LeftBound171059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171059.derived selector witness)

def rawBound : CoeffClass := LeftBound171059.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound171059.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound171063

namespace LeftBound171080
def owner : Owner := ⟨.program ⟨257⟩, ⟨23222⟩⟩
def transferEvent : Nat := 171080
def frameStart : Nat := 171030
def rule : BoundRule := .sum [.predecessor 0 171078 .coefficient, .predecessor 1 171079 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171078 .coefficient)
      LeftBound171063.bound (LeftBound171063.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound171063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171079 .coefficient)
      LeftAuthority171076.bound (LeftAuthority171076.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority171076.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171063.bound, LeftAuthority171076.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171063.bound, LeftAuthority171076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171063.actual selector witness, LeftAuthority171076.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171080

namespace LeftBound171083
def owner : Owner := ⟨.program ⟨257⟩, ⟨23223⟩⟩
def transferEvent : Nat := 171083
def frameStart : Nat := 171030
def rule : BoundRule := .identity (.predecessor 0 171082 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171082 .coefficient)
      LeftBound171080.bound (LeftBound171080.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound171080.derived selector witness)

def rawBound : CoeffClass := LeftBound171080.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound171080.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound171083

namespace LeftBound171089
def owner : Owner := ⟨.program ⟨257⟩, ⟨23224⟩⟩
def transferEvent : Nat := 171089
def frameStart : Nat := 171030
def rule : BoundRule := .product (.predecessor 0 171087 .coefficient) (.predecessor 1 171088 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171087 .coefficient)
      LeftAuthority171085.bound (LeftAuthority171085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171088 .coefficient)
      LeftBound171083.bound (LeftBound171083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority171085.bound LeftBound171083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171085.bound, LeftBound171083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority171085.actual selector witness) * (LeftBound171083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171089

namespace LeftBound171105
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 171105
def frameStart : Nat := 171030
def rule : BoundRule := .scale (.predecessor 0 171103 .coefficient) (.value (.predecessor 1 171104 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171103 .coefficient)
      LeftAuthority171101.bound (LeftAuthority171101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171104 .coefficient)
      LeftAuthority171092.bound (LeftAuthority171092.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority171092.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority171101.bound LeftAuthority171092.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171101.bound, LeftAuthority171092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority171101.actual selector witness) * (LeftAuthority171092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound171105

namespace LeftBound171108
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 171108
def frameStart : Nat := 171030
def rule : BoundRule := .identity (.predecessor 0 171107 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171107 .coefficient)
      LeftAuthority171095.bound (LeftAuthority171095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171095.derived selector witness)

def rawBound : CoeffClass := LeftAuthority171095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority171095.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound171108

namespace LeftBound171112
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 171112
def frameStart : Nat := 171030
def rule : BoundRule := .product (.predecessor 0 171110 .coefficient) (.predecessor 1 171111 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171110 .coefficient)
      LeftBound171108.bound (LeftBound171108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171111 .coefficient)
      LeftBound171105.bound (LeftBound171105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound171108.bound LeftBound171105.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171108.bound, LeftBound171105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound171108.actual selector witness) * (LeftBound171105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171112

namespace LeftBound171117
def owner : Owner := ⟨.program ⟨257⟩, ⟨23225⟩⟩
def transferEvent : Nat := 171117
def frameStart : Nat := 171030
def rule : BoundRule := .sum [.predecessor 0 171115 .coefficient, .predecessor 1 171116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171115 .coefficient)
      LeftBound171112.bound (LeftBound171112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171116 .coefficient)
      LeftBound171089.bound (LeftBound171089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171089.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171112.bound, LeftBound171089.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171112.bound, LeftBound171089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171112.actual selector witness, LeftBound171089.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171117

namespace LeftBound171121
def owner : Owner := ⟨.program ⟨257⟩, ⟨23486⟩⟩
def transferEvent : Nat := 171121
def frameStart : Nat := 171030
def rule : BoundRule := .product (.predecessor 0 171119 .coefficient) (.predecessor 1 171120 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171119 .coefficient)
      LeftBound171117.bound (LeftBound171117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171120 .coefficient)
      LeftAuthority171074.bound (LeftAuthority171074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound171117.bound LeftAuthority171074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171117.bound, LeftAuthority171074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound171117.actual selector witness) * (LeftAuthority171074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171121

namespace LeftBound171132
def owner : Owner := ⟨.program ⟨257⟩, ⟨21842⟩⟩
def transferEvent : Nat := 171132
def frameStart : Nat := 171030
def rule : BoundRule := .product (.predecessor 0 171130 .coefficient) (.predecessor 1 171131 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171130 .coefficient)
      LeftAuthority171085.bound (LeftAuthority171085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171131 .coefficient)
      LeftAuthority171128.bound (LeftAuthority171128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority171085.bound LeftAuthority171128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171085.bound, LeftAuthority171128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority171085.actual selector witness) * (LeftAuthority171128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171132

namespace LeftBound171140
def owner : Owner := ⟨.program ⟨257⟩, ⟨21843⟩⟩
def transferEvent : Nat := 171140
def frameStart : Nat := 171030
def rule : BoundRule := .sum [.predecessor 0 171138 .coefficient, .predecessor 1 171139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171138 .coefficient)
      LeftAuthority171136.bound (LeftAuthority171136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority171136.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority171136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171139 .coefficient)
      LeftBound171132.bound (LeftBound171132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171132.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority171136.bound, LeftBound171132.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority171136.bound, LeftBound171132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority171136.actual selector witness, LeftBound171132.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171140

namespace LeftBound171144
def owner : Owner := ⟨.program ⟨257⟩, ⟨23487⟩⟩
def transferEvent : Nat := 171144
def frameStart : Nat := 171030
def rule : BoundRule := .sum [.predecessor 0 171142 .coefficient, .predecessor 1 171143 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171142 .coefficient)
      LeftBound171140.bound (LeftBound171140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171143 .coefficient)
      LeftBound171121.bound (LeftBound171121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound171140.bound, LeftBound171121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171140.bound, LeftBound171121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound171140.actual selector witness, LeftBound171121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171144

namespace LeftBound171157
def owner : Owner := ⟨.program ⟨257⟩, ⟨23485⟩⟩
def transferEvent : Nat := 171157
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 171155 .coefficient, .predecessor 1 171156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171155 .coefficient)
      LeftBound170978.bound (LeftBound170978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound170978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound170978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171156 .coefficient)
      LeftBound170961.bound (LeftBound170961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events667.exact170968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound170961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound170961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound170978.bound, LeftBound170961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound170978.bound, LeftBound170961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound170978.actual selector witness, LeftBound170961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171157

namespace LeftBound171160
def owner : Owner := ⟨.program ⟨257⟩, ⟨23485⟩⟩
def transferEvent : Nat := 171160
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 171154 .summary, .result 170968 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 171154 .summary)
      LeftBound170980.bound (LeftBound170980.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22412⟩⟩) (rawTerms := some (Proof.Events668.exact171154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound170980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 170968 .summary)
      LeftBound170963.bound (LeftBound170963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23484⟩⟩) (rawTerms := some (Proof.Events667.exact170968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound170963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound170980.bound, LeftBound170963.bound]
def bound : CoeffClass := .finite ⟨2997834576566628384768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound170980.bound, LeftBound170963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound170980.actual selector witness, LeftBound170963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound171160

namespace LeftBound171164
def owner : Owner := ⟨.program ⟨257⟩, ⟨23998⟩⟩
def transferEvent : Nat := 171164
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 171162 .coefficient) (.predecessor 1 171163 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 171162 .coefficient)
      LeftBound171157.bound (LeftBound171157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events668.exact171161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound171157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound171157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 171163 .coefficient)
      LeftAuthority170883.bound (LeftAuthority170883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events667.exact170884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority170883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority170883.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound171157.bound LeftAuthority170883.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound171157.bound, LeftAuthority170883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound171157.actual selector witness) * (LeftAuthority170883.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound171164

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
