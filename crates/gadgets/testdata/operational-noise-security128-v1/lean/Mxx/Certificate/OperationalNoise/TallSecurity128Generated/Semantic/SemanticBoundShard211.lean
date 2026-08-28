import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard107
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard175
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard210

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound36900
def owner : Owner := ⟨.program ⟨257⟩, ⟨59733⟩⟩
def transferEvent : Nat := 36900
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36898 .coefficient, .predecessor 1 36899 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36898 .coefficient)
      LeftBound36895.bound (LeftBound36895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36899 .coefficient)
      LeftBound36890.bound (LeftBound36890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36895.bound, LeftBound36890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36895.bound, LeftBound36890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound36895.actual selector witness, LeftBound36890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36900

namespace LeftBound36904
def owner : Owner := ⟨.program ⟨257⟩, ⟨59734⟩⟩
def transferEvent : Nat := 36904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36902 .coefficient, .predecessor 1 36903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36902 .coefficient)
      LeftBound36900.bound (LeftBound36900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36903 .coefficient)
      LeftBound22122.bound (LeftBound22122.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22122.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36900.bound, LeftBound22122.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36900.bound, LeftBound22122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound36900.actual selector witness, LeftBound22122.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36904

namespace LeftBound36905
def owner : Owner := ⟨.program ⟨257⟩, ⟨59734⟩⟩
def transferEvent : Nat := 36905
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨117⟩⟩]⟩ [⟨.result 22123 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22123 .coefficient)
      LeftBound22122.bound (LeftBound22122.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨117⟩⟩) (rawTerms := some (Proof.Events086.exact22123RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22122.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22122.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound22122.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22122.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound22122.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36905

namespace LeftBound36910
def owner : Owner := ⟨.program ⟨257⟩, ⟨59735⟩⟩
def transferEvent : Nat := 36910
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36908 .coefficient) (.predecessor 1 36909 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36908 .coefficient)
      LeftBound36904.bound (LeftBound36904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36909 .coefficient)
      LeftBound22119.bound (LeftBound22119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events086.exact22120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22119.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36904.bound LeftBound22119.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36904.bound, LeftBound22119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36904.actual selector witness) * (LeftBound22119.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36910

namespace LeftBound36911
def owner : Owner := ⟨.program ⟨257⟩, ⟨59735⟩⟩
def transferEvent : Nat := 36911
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9535⟩⟩]⟩ [⟨.result 22116 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 22116 .coefficient)
      LeftAuthority22115.bound (LeftAuthority22115.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9535⟩⟩) (rawTerms := some (Proof.Events086.exact22116RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22115.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22115.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority22115.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36911

namespace LeftBound36912
def owner : Owner := ⟨.program ⟨257⟩, ⟨59735⟩⟩
def transferEvent : Nat := 36912
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36907 .summary) (.transfer 36911) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36907 .summary)
      LeftBound36905.bound (LeftBound36905.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59734⟩⟩) (rawTerms := some (Proof.Events144.exact36907RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 36911)
      LeftBound36911.bound (LeftBound36911.actual selector witness) := by
  exact .transfer (LeftBound36911.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36905.bound LeftBound36911.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36905.bound, LeftBound36911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36905.actual selector witness) * (LeftBound36911.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36912

namespace LeftBound36920
def owner : Owner := ⟨.program ⟨257⟩, ⟨59736⟩⟩
def transferEvent : Nat := 36920
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 36918 .coefficient, .predecessor 1 36919 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36918 .coefficient)
      LeftBound36910.bound (LeftBound36910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36919 .coefficient)
      LeftBound36882.bound (LeftBound36882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36910.bound, LeftBound36882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36910.bound, LeftBound36882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound36910.actual selector witness, LeftBound36882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36920

namespace LeftBound36922
def owner : Owner := ⟨.program ⟨257⟩, ⟨59736⟩⟩
def transferEvent : Nat := 36922
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 36917 .summary, .result 36887 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36917 .summary)
      LeftBound36912.bound (LeftBound36912.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59735⟩⟩) (rawTerms := some (Proof.Events144.exact36917RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36887 .summary)
      LeftBound36884.bound (LeftBound36884.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59731⟩⟩) (rawTerms := some (Proof.Events144.exact36887RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound36912.bound, LeftBound36884.bound]
def bound : CoeffClass := .finite ⟨279188209664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36912.bound, LeftBound36884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound36912.actual selector witness, LeftBound36884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound36922

namespace LeftBound36926
def owner : Owner := ⟨.program ⟨257⟩, ⟨61559⟩⟩
def transferEvent : Nat := 36926
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36924 .coefficient) (.predecessor 1 36925 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36924 .coefficient)
      LeftBound36920.bound (LeftBound36920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36925 .coefficient)
      LeftAuthority36858.bound (LeftAuthority36858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events143.exact36859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36920.bound LeftAuthority36858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36920.bound, LeftAuthority36858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36920.actual selector witness) * (LeftAuthority36858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36926

namespace LeftBound36927
def owner : Owner := ⟨.program ⟨257⟩, ⟨61559⟩⟩
def transferEvent : Nat := 36927
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨61558⟩⟩]⟩ [⟨.result 36859 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36859 .coefficient)
      LeftAuthority36858.bound (LeftAuthority36858.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨61558⟩⟩) (rawTerms := some (Proof.Events143.exact36859RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36858.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36858.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority36858.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36927

namespace LeftBound36928
def owner : Owner := ⟨.program ⟨257⟩, ⟨61559⟩⟩
def transferEvent : Nat := 36928
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36923 .summary) (.transfer 36927) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36923 .summary)
      LeftBound36922.bound (LeftBound36922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59736⟩⟩) (rawTerms := some (Proof.Events144.exact36923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 36927)
      LeftBound36927.bound (LeftBound36927.actual selector witness) := by
  exact .transfer (LeftBound36927.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound36922.bound LeftBound36927.bound
def bound : CoeffClass := .finite ⟨2997760574839177871360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36922.bound, LeftBound36927.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound36922.actual selector witness) * (LeftBound36927.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36928

namespace LeftBound36939
def owner : Owner := ⟨.program ⟨257⟩, ⟨60481⟩⟩
def transferEvent : Nat := 36939
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 36937 .coefficient) (.value (.predecessor 1 36938 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36937 .coefficient)
      LeftAuthority36935.bound (LeftAuthority36935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36935.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36938 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority36935.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36935.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority36935.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound36939

namespace LeftBound36943
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def transferEvent : Nat := 36943
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 36941 .coefficient) (.predecessor 1 36942 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 36941 .coefficient)
      LeftBound32117.bound (LeftBound32117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound32117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound32117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 36942 .coefficient)
      LeftBound36939.bound (LeftBound36939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact36940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36939.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32117.bound LeftBound36939.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32117.bound, LeftBound36939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32117.actual selector witness) * (LeftBound36939.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36943

namespace LeftBound36944
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def transferEvent : Nat := 36944
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨60479⟩⟩]⟩ [⟨.result 36936 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 36936 .coefficient)
      LeftAuthority36935.bound (LeftAuthority36935.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨60479⟩⟩) (rawTerms := some (Proof.Events144.exact36936RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority36935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority36935.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority36935.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority36935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority36935.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound36944

namespace LeftBound36945
def owner : Owner := ⟨.program ⟨257⟩, ⟨60482⟩⟩
def transferEvent : Nat := 36945
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 32120 .summary) (.transfer 36944) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 32120 .summary)
      LeftBound32118.bound (LeftBound32118.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11643⟩⟩) (rawTerms := some (Proof.Events125.exact32120RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound32118.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 36944)
      LeftBound36944.bound (LeftBound36944.actual selector witness) := by
  exact .transfer (LeftBound36944.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound32118.bound LeftBound36944.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound32118.bound, LeftBound36944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound32118.actual selector witness) * (LeftBound36944.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound36945

namespace LeftBound37024
def owner : Owner := ⟨.program ⟨257⟩, ⟨59729⟩⟩
def transferEvent : Nat := 37024
def frameStart : Nat := 36995
def rule : BoundRule := .product (.predecessor 0 37022 .coefficient) (.predecessor 1 37023 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 37022 .coefficient)
      LeftAuthority37020.bound (LeftAuthority37020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37020.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 37023 .coefficient)
      LeftAuthority37017.bound (LeftAuthority37017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37017.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37020.bound LeftAuthority37017.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37020.bound, LeftAuthority37017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority37020.actual selector witness) * (LeftAuthority37017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37024

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
