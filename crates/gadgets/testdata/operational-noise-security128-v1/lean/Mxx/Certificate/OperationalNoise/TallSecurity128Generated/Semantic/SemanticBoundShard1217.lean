import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1188
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1216

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound182073
def owner : Owner := ⟨.program ⟨257⟩, ⟨27780⟩⟩
def transferEvent : Nat := 182073
def frameStart : Nat := 182008
def rule : BoundRule := .product (.predecessor 0 182071 .coefficient) (.predecessor 1 182072 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182071 .coefficient)
      LeftAuthority182069.bound (LeftAuthority182069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182069.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182072 .coefficient)
      LeftBound182067.bound (LeftBound182067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182067.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority182069.bound LeftBound182067.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority182069.bound, LeftBound182067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority182069.actual selector witness) * (LeftBound182067.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182073

namespace LeftBound182081
def owner : Owner := ⟨.program ⟨257⟩, ⟨27781⟩⟩
def transferEvent : Nat := 182081
def frameStart : Nat := 182008
def rule : BoundRule := .sum [.predecessor 0 182079 .coefficient, .predecessor 1 182080 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182079 .coefficient)
      LeftAuthority182077.bound (LeftAuthority182077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182077.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182080 .coefficient)
      LeftBound182073.bound (LeftBound182073.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182073.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182073.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority182077.bound, LeftBound182073.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority182077.bound, LeftBound182073.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority182077.actual selector witness, LeftBound182073.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182081

namespace LeftBound182085
def owner : Owner := ⟨.program ⟨257⟩, ⟨28365⟩⟩
def transferEvent : Nat := 182085
def frameStart : Nat := 182008
def rule : BoundRule := .product (.predecessor 0 182083 .coefficient) (.predecessor 1 182084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182083 .coefficient)
      LeftBound182081.bound (LeftBound182081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182084 .coefficient)
      LeftAuthority182058.bound (LeftAuthority182058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182058.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound182081.bound LeftAuthority182058.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182081.bound, LeftAuthority182058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound182081.actual selector witness) * (LeftAuthority182058.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182085

namespace LeftBound182096
def owner : Owner := ⟨.program ⟨257⟩, ⟨26659⟩⟩
def transferEvent : Nat := 182096
def frameStart : Nat := 182008
def rule : BoundRule := .product (.predecessor 0 182094 .coefficient) (.predecessor 1 182095 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182094 .coefficient)
      LeftAuthority182069.bound (LeftAuthority182069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182070RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182069.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182069.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182095 .coefficient)
      LeftAuthority182092.bound (LeftAuthority182092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority182069.bound LeftAuthority182092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority182069.bound, LeftAuthority182092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority182069.actual selector witness) * (LeftAuthority182092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182096

namespace LeftBound182104
def owner : Owner := ⟨.program ⟨257⟩, ⟨26660⟩⟩
def transferEvent : Nat := 182104
def frameStart : Nat := 182008
def rule : BoundRule := .sum [.predecessor 0 182102 .coefficient, .predecessor 1 182103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182102 .coefficient)
      LeftAuthority182100.bound (LeftAuthority182100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority182100.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority182100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182103 .coefficient)
      LeftBound182096.bound (LeftBound182096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182096.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority182100.bound, LeftBound182096.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority182100.bound, LeftBound182096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority182100.actual selector witness, LeftBound182096.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182104

namespace LeftBound182108
def owner : Owner := ⟨.program ⟨257⟩, ⟨28368⟩⟩
def transferEvent : Nat := 182108
def frameStart : Nat := 182008
def rule : BoundRule := .sum [.predecessor 0 182106 .coefficient, .predecessor 1 182107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182106 .coefficient)
      LeftBound182104.bound (LeftBound182104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182107 .coefficient)
      LeftBound182085.bound (LeftBound182085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound182104.bound, LeftBound182085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182104.bound, LeftBound182085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound182104.actual selector witness, LeftBound182085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182108

namespace LeftBound182121
def owner : Owner := ⟨.program ⟨257⟩, ⟨28367⟩⟩
def transferEvent : Nat := 182121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 182119 .coefficient, .predecessor 1 182120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182119 .coefficient)
      LeftBound181950.bound (LeftBound181950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182120 .coefficient)
      LeftBound181933.bound (LeftBound181933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events710.exact181940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound181933.bound, RecordedBoundRefines] <;> decide)
      (LeftBound181933.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound181950.bound, LeftBound181933.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181950.bound, LeftBound181933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound181950.actual selector witness, LeftBound181933.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182121

namespace LeftBound182124
def owner : Owner := ⟨.program ⟨257⟩, ⟨28367⟩⟩
def transferEvent : Nat := 182124
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 182118 .summary, .result 181940 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 182118 .summary)
      LeftBound181952.bound (LeftBound181952.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27219⟩⟩) (rawTerms := some (Proof.Events711.exact182118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound181952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 181940 .summary)
      LeftBound181935.bound (LeftBound181935.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28366⟩⟩) (rawTerms := some (Proof.Events710.exact181940RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound181935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound181952.bound, LeftBound181935.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound181952.bound, LeftBound181935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound181952.actual selector witness, LeftBound181935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182124

namespace LeftBound182148
def owner : Owner := ⟨.program ⟨257⟩, ⟨25767⟩⟩
def transferEvent : Nat := 182148
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 182146 .coefficient) (.predecessor 1 182147 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182146 .coefficient)
      LeftAuthority8505.bound (LeftAuthority8505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182147 .coefficient)
      LeftBound178276.bound (LeftBound178276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority8505.bound LeftBound178276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8505.bound, LeftBound178276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority8505.actual selector witness) * (LeftBound178276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound182148

namespace LeftBound182153
def owner : Owner := ⟨.program ⟨257⟩, ⟨8924⟩⟩
def transferEvent : Nat := 182153
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 182151 .coefficient) (.predecessor 1 182152 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182151 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182152 .coefficient)
      LeftBound21087.bound (LeftBound21087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftBound21087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftBound21087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftBound21087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182153

namespace LeftBound182158
def owner : Owner := ⟨.program ⟨257⟩, ⟨25768⟩⟩
def transferEvent : Nat := 182158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 182156 .coefficient, .predecessor 1 182157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182156 .coefficient)
      LeftBound182153.bound (LeftBound182153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182157 .coefficient)
      LeftBound182148.bound (LeftBound182148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound182153.bound, LeftBound182148.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182153.bound, LeftBound182148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound182153.actual selector witness, LeftBound182148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182158

namespace LeftBound182162
def owner : Owner := ⟨.program ⟨257⟩, ⟨25769⟩⟩
def transferEvent : Nat := 182162
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 182160 .coefficient, .predecessor 1 182161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182160 .coefficient)
      LeftBound182158.bound (LeftBound182158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182161 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound182158.bound, LeftBound21079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182158.bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound182158.actual selector witness, LeftBound21079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound182162

namespace LeftBound182163
def owner : Owner := ⟨.program ⟨257⟩, ⟨25769⟩⟩
def transferEvent : Nat := 182163
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩ [⟨.result 21080 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21080 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21079.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21079.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound182163

namespace LeftBound182168
def owner : Owner := ⟨.program ⟨257⟩, ⟨65529⟩⟩
def transferEvent : Nat := 182168
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 182166 .coefficient) (.predecessor 1 182167 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 182166 .coefficient)
      LeftBound182162.bound (LeftBound182162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events711.exact182165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound182162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound182162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 182167 .coefficient)
      LeftAuthority8508.bound (LeftAuthority8508.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8508.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound182162.bound LeftAuthority8508.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182162.bound, LeftAuthority8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound182162.actual selector witness) * (LeftAuthority8508.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182168

namespace LeftBound182169
def owner : Owner := ⟨.program ⟨257⟩, ⟨65529⟩⟩
def transferEvent : Nat := 182169
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨65526⟩⟩], []⟩ [⟨.result 8509 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 8509 .coefficient)
      LeftAuthority8508.bound (LeftAuthority8508.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨65526⟩⟩) (rawTerms := some (Proof.Events033.exact8509RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8508.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8508.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8508.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8508.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority8508.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound182169

namespace LeftBound182170
def owner : Owner := ⟨.program ⟨257⟩, ⟨65529⟩⟩
def transferEvent : Nat := 182170
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 182165 .summary) (.transfer 182169) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 182165 .summary)
      LeftBound182163.bound (LeftBound182163.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25769⟩⟩) (rawTerms := some (Proof.Events711.exact182165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound182163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 182169)
      LeftBound182169.bound (LeftBound182169.actual selector witness) := by
  exact .transfer (LeftBound182169.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound182163.bound LeftBound182169.bound
def bound : CoeffClass := .finite ⟨23855104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound182163.bound, LeftBound182169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound182163.actual selector witness) * (LeftBound182169.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound182170

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
