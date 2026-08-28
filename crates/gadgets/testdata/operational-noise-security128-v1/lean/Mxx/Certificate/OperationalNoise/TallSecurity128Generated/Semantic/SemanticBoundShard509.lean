import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard508

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79935
def owner : Owner := ⟨.program ⟨257⟩, ⟨65608⟩⟩
def transferEvent : Nat := 79935
def frameStart : Nat := 79906
def rule : BoundRule := .product (.predecessor 0 79933 .coefficient) (.predecessor 1 79934 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79933 .coefficient)
      LeftAuthority79931.bound (LeftAuthority79931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79934 .coefficient)
      LeftAuthority79928.bound (LeftAuthority79928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79928.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority79931.bound LeftAuthority79928.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79931.bound, LeftAuthority79928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority79931.actual selector witness) * (LeftAuthority79928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79935

namespace LeftBound79939
def owner : Owner := ⟨.program ⟨257⟩, ⟨65609⟩⟩
def transferEvent : Nat := 79939
def frameStart : Nat := 79906
def rule : BoundRule := .identity (.predecessor 0 79938 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79938 .coefficient)
      LeftBound79935.bound (LeftBound79935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79935.derived selector witness)

def rawBound : CoeffClass := LeftBound79935.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound79935.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79939

namespace LeftBound79956
def owner : Owner := ⟨.program ⟨257⟩, ⟨68951⟩⟩
def transferEvent : Nat := 79956
def frameStart : Nat := 79906
def rule : BoundRule := .sum [.predecessor 0 79954 .coefficient, .predecessor 1 79955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79954 .coefficient)
      LeftBound79939.bound (LeftBound79939.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79955 .coefficient)
      LeftAuthority79952.bound (LeftAuthority79952.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority79952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79939.bound, LeftAuthority79952.bound]
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79939.bound, LeftAuthority79952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79939.actual selector witness, LeftAuthority79952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79956

namespace LeftBound79959
def owner : Owner := ⟨.program ⟨257⟩, ⟨68952⟩⟩
def transferEvent : Nat := 79959
def frameStart : Nat := 79906
def rule : BoundRule := .identity (.predecessor 0 79958 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79958 .coefficient)
      LeftBound79956.bound (LeftBound79956.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound79956.derived selector witness)

def rawBound : CoeffClass := LeftBound79956.bound
def bound : CoeffClass := .finite ⟨784, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound79956.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79959

namespace LeftBound79965
def owner : Owner := ⟨.program ⟨257⟩, ⟨68953⟩⟩
def transferEvent : Nat := 79965
def frameStart : Nat := 79906
def rule : BoundRule := .product (.predecessor 0 79963 .coefficient) (.predecessor 1 79964 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79963 .coefficient)
      LeftAuthority79961.bound (LeftAuthority79961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79964 .coefficient)
      LeftBound79959.bound (LeftBound79959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79959.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority79961.bound LeftBound79959.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79961.bound, LeftBound79959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority79961.actual selector witness) * (LeftBound79959.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79965

namespace LeftBound79981
def owner : Owner := ⟨.program ⟨257⟩, ⟨9542⟩⟩
def transferEvent : Nat := 79981
def frameStart : Nat := 79906
def rule : BoundRule := .scale (.predecessor 0 79979 .coefficient) (.value (.predecessor 1 79980 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79979 .coefficient)
      LeftAuthority79977.bound (LeftAuthority79977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79977.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79980 .coefficient)
      LeftAuthority79968.bound (LeftAuthority79968.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority79968.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority79977.bound LeftAuthority79968.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79977.bound, LeftAuthority79968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority79977.actual selector witness) * (LeftAuthority79968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79981

namespace LeftBound79984
def owner : Owner := ⟨.program ⟨257⟩, ⟨7294⟩⟩
def transferEvent : Nat := 79984
def frameStart : Nat := 79906
def rule : BoundRule := .identity (.predecessor 0 79983 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79983 .coefficient)
      LeftAuthority79971.bound (LeftAuthority79971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79971.derived selector witness)

def rawBound : CoeffClass := LeftAuthority79971.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority79971.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound79984

namespace LeftBound79988
def owner : Owner := ⟨.program ⟨257⟩, ⟨9543⟩⟩
def transferEvent : Nat := 79988
def frameStart : Nat := 79906
def rule : BoundRule := .product (.predecessor 0 79986 .coefficient) (.predecessor 1 79987 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79986 .coefficient)
      LeftBound79984.bound (LeftBound79984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79987 .coefficient)
      LeftBound79981.bound (LeftBound79981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79981.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound79984.bound LeftBound79981.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79984.bound, LeftBound79981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound79984.actual selector witness) * (LeftBound79981.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79988

namespace LeftBound79993
def owner : Owner := ⟨.program ⟨257⟩, ⟨68954⟩⟩
def transferEvent : Nat := 79993
def frameStart : Nat := 79906
def rule : BoundRule := .sum [.predecessor 0 79991 .coefficient, .predecessor 1 79992 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79991 .coefficient)
      LeftBound79988.bound (LeftBound79988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79992 .coefficient)
      LeftBound79965.bound (LeftBound79965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79965.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79988.bound, LeftBound79965.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79988.bound, LeftBound79965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79988.actual selector witness, LeftBound79965.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79993

namespace LeftBound79997
def owner : Owner := ⟨.program ⟨257⟩, ⟨69309⟩⟩
def transferEvent : Nat := 79997
def frameStart : Nat := 79906
def rule : BoundRule := .product (.predecessor 0 79995 .coefficient) (.predecessor 1 79996 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 79995 .coefficient)
      LeftBound79993.bound (LeftBound79993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79994RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 79996 .coefficient)
      LeftAuthority79950.bound (LeftAuthority79950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79950.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound79993.bound LeftAuthority79950.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79993.bound, LeftAuthority79950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound79993.actual selector witness) * (LeftAuthority79950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79997

namespace LeftBound80008
def owner : Owner := ⟨.program ⟨257⟩, ⟨65838⟩⟩
def transferEvent : Nat := 80008
def frameStart : Nat := 79906
def rule : BoundRule := .product (.predecessor 0 80006 .coefficient) (.predecessor 1 80007 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 80006 .coefficient)
      LeftAuthority79961.bound (LeftAuthority79961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 80007 .coefficient)
      LeftAuthority80004.bound (LeftAuthority80004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80004.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority79961.bound LeftAuthority80004.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79961.bound, LeftAuthority80004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority79961.actual selector witness) * (LeftAuthority80004.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80008

namespace LeftBound80016
def owner : Owner := ⟨.program ⟨257⟩, ⟨65839⟩⟩
def transferEvent : Nat := 80016
def frameStart : Nat := 79906
def rule : BoundRule := .sum [.predecessor 0 80014 .coefficient, .predecessor 1 80015 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 80014 .coefficient)
      LeftAuthority80012.bound (LeftAuthority80012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority80012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority80012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 80015 .coefficient)
      LeftBound80008.bound (LeftBound80008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80008.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority80012.bound, LeftBound80008.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority80012.bound, LeftBound80008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority80012.actual selector witness, LeftBound80008.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80016

namespace LeftBound80020
def owner : Owner := ⟨.program ⟨257⟩, ⟨69310⟩⟩
def transferEvent : Nat := 80020
def frameStart : Nat := 79906
def rule : BoundRule := .sum [.predecessor 0 80018 .coefficient, .predecessor 1 80019 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 80018 .coefficient)
      LeftBound80016.bound (LeftBound80016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 80019 .coefficient)
      LeftBound79997.bound (LeftBound79997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79997.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80016.bound, LeftBound79997.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80016.bound, LeftBound79997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound80016.actual selector witness, LeftBound79997.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80020

namespace LeftBound80033
def owner : Owner := ⟨.program ⟨257⟩, ⟨69308⟩⟩
def transferEvent : Nat := 80033
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80031 .coefficient, .predecessor 1 80032 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 80031 .coefficient)
      LeftBound79854.bound (LeftBound79854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79854.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 80032 .coefficient)
      LeftBound79837.bound (LeftBound79837.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79837.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79854.bound, LeftBound79837.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79854.bound, LeftBound79837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79854.actual selector witness, LeftBound79837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80033

namespace LeftBound80036
def owner : Owner := ⟨.program ⟨257⟩, ⟨69308⟩⟩
def transferEvent : Nat := 80036
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 80030 .summary, .result 79844 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 80030 .summary)
      LeftBound79856.bound (LeftBound79856.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨67833⟩⟩) (rawTerms := some (Proof.Events312.exact80030RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79856.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 79844 .summary)
      LeftBound79839.bound (LeftBound79839.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69307⟩⟩) (rawTerms := some (Proof.Events311.exact79844RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79856.bound, LeftBound79839.bound]
def bound : CoeffClass := .finite ⟨2998054127048462696448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79856.bound, LeftBound79839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound79856.actual selector witness, LeftBound79839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80036

namespace LeftBound80040
def owner : Owner := ⟨.program ⟨257⟩, ⟨70653⟩⟩
def transferEvent : Nat := 80040
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80038 .coefficient) (.predecessor 1 80039 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 80038 .coefficient)
      LeftBound80033.bound (LeftBound80033.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80033.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 80039 .coefficient)
      LeftAuthority79759.bound (LeftAuthority79759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79759.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound80033.bound LeftAuthority79759.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80033.bound, LeftAuthority79759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound80033.actual selector witness) * (LeftAuthority79759.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80040

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
