import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1185
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1188
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1245

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound185929
def owner : Owner := ⟨.program ⟨257⟩, ⟨23300⟩⟩
def transferEvent : Nat := 185929
def frameStart : Nat := 185864
def rule : BoundRule := .product (.predecessor 0 185927 .coefficient) (.predecessor 1 185928 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185927 .coefficient)
      LeftAuthority185925.bound (LeftAuthority185925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185925.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185928 .coefficient)
      LeftBound185923.bound (LeftBound185923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185923.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority185925.bound LeftBound185923.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority185925.bound, LeftBound185923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority185925.actual selector witness) * (LeftBound185923.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound185929

namespace LeftBound185937
def owner : Owner := ⟨.program ⟨257⟩, ⟨23301⟩⟩
def transferEvent : Nat := 185937
def frameStart : Nat := 185864
def rule : BoundRule := .sum [.predecessor 0 185935 .coefficient, .predecessor 1 185936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185935 .coefficient)
      LeftAuthority185933.bound (LeftAuthority185933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185933.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185936 .coefficient)
      LeftBound185929.bound (LeftBound185929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority185933.bound, LeftBound185929.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority185933.bound, LeftBound185929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority185933.actual selector witness, LeftBound185929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound185937

namespace LeftBound185941
def owner : Owner := ⟨.program ⟨257⟩, ⟨23966⟩⟩
def transferEvent : Nat := 185941
def frameStart : Nat := 185864
def rule : BoundRule := .product (.predecessor 0 185939 .coefficient) (.predecessor 1 185940 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185939 .coefficient)
      LeftBound185937.bound (LeftBound185937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185937.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185940 .coefficient)
      LeftAuthority185914.bound (LeftAuthority185914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound185937.bound LeftAuthority185914.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185937.bound, LeftAuthority185914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound185937.actual selector witness) * (LeftAuthority185914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound185941

namespace LeftBound185952
def owner : Owner := ⟨.program ⟨257⟩, ⟨22145⟩⟩
def transferEvent : Nat := 185952
def frameStart : Nat := 185864
def rule : BoundRule := .product (.predecessor 0 185950 .coefficient) (.predecessor 1 185951 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185950 .coefficient)
      LeftAuthority185925.bound (LeftAuthority185925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185925.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185951 .coefficient)
      LeftAuthority185948.bound (LeftAuthority185948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185948.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185948.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority185925.bound LeftAuthority185948.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority185925.bound, LeftAuthority185948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority185925.actual selector witness) * (LeftAuthority185948.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound185952

namespace LeftBound185960
def owner : Owner := ⟨.program ⟨257⟩, ⟨22146⟩⟩
def transferEvent : Nat := 185960
def frameStart : Nat := 185864
def rule : BoundRule := .sum [.predecessor 0 185958 .coefficient, .predecessor 1 185959 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185958 .coefficient)
      LeftAuthority185956.bound (LeftAuthority185956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority185956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority185956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185959 .coefficient)
      LeftBound185952.bound (LeftBound185952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority185956.bound, LeftBound185952.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority185956.bound, LeftBound185952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority185956.actual selector witness, LeftBound185952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound185960

namespace LeftBound185964
def owner : Owner := ⟨.program ⟨257⟩, ⟨23970⟩⟩
def transferEvent : Nat := 185964
def frameStart : Nat := 185864
def rule : BoundRule := .sum [.predecessor 0 185962 .coefficient, .predecessor 1 185963 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185962 .coefficient)
      LeftBound185960.bound (LeftBound185960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185963 .coefficient)
      LeftBound185941.bound (LeftBound185941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185941.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound185960.bound, LeftBound185941.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185960.bound, LeftBound185941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound185960.actual selector witness, LeftBound185941.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound185964

namespace LeftBound185977
def owner : Owner := ⟨.program ⟨257⟩, ⟨23968⟩⟩
def transferEvent : Nat := 185977
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 185975 .coefficient, .predecessor 1 185976 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 185975 .coefficient)
      LeftBound185806.bound (LeftBound185806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact185974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185806.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 185976 .coefficient)
      LeftBound185789.bound (LeftBound185789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events725.exact185796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound185789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound185789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound185806.bound, LeftBound185789.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185806.bound, LeftBound185789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound185806.actual selector witness, LeftBound185789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound185977

namespace LeftBound185980
def owner : Owner := ⟨.program ⟨257⟩, ⟨23968⟩⟩
def transferEvent : Nat := 185980
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 185974 .summary, .result 185796 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185974 .summary)
      LeftBound185808.bound (LeftBound185808.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22739⟩⟩) (rawTerms := some (Proof.Events726.exact185974RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 185796 .summary)
      LeftBound185791.bound (LeftBound185791.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23967⟩⟩) (rawTerms := some (Proof.Events725.exact185796RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound185791.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound185808.bound, LeftBound185791.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound185808.bound, LeftBound185791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound185808.actual selector witness, LeftBound185791.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound185980

namespace LeftBound186004
def owner : Owner := ⟨.program ⟨257⟩, ⟨18349⟩⟩
def transferEvent : Nat := 186004
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 186002 .coefficient) (.predecessor 1 186003 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186002 .coefficient)
      LeftAuthority8689.bound (LeftAuthority8689.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8690RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8689.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8689.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186003 .coefficient)
      LeftBound178276.bound (LeftBound178276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority8689.bound LeftBound178276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8689.bound, LeftBound178276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority8689.actual selector witness) * (LeftBound178276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound186004

namespace LeftBound186009
def owner : Owner := ⟨.program ⟨257⟩, ⟨8953⟩⟩
def transferEvent : Nat := 186009
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 186007 .coefficient) (.predecessor 1 186008 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186007 .coefficient)
      LeftBound178147.bound (LeftBound178147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events695.exact178148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186008 .coefficient)
      LeftBound25095.bound (LeftBound25095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound178147.bound LeftBound25095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178147.bound, LeftBound25095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound178147.actual selector witness) * (LeftBound25095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186009

namespace LeftBound186014
def owner : Owner := ⟨.program ⟨257⟩, ⟨18350⟩⟩
def transferEvent : Nat := 186014
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186012 .coefficient, .predecessor 1 186013 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186012 .coefficient)
      LeftBound186009.bound (LeftBound186009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact186011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186013 .coefficient)
      LeftBound186004.bound (LeftBound186004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact186006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186004.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186009.bound, LeftBound186004.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186009.bound, LeftBound186004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186009.actual selector witness, LeftBound186004.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186014

namespace LeftBound186018
def owner : Owner := ⟨.program ⟨257⟩, ⟨18351⟩⟩
def transferEvent : Nat := 186018
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 186016 .coefficient, .predecessor 1 186017 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186016 .coefficient)
      LeftBound186014.bound (LeftBound186014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact186015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186014.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186017 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound186014.bound, LeftBound25087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186014.bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound186014.actual selector witness, LeftBound25087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound186018

namespace LeftBound186019
def owner : Owner := ⟨.program ⟨257⟩, ⟨18351⟩⟩
def transferEvent : Nat := 186019
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩ [⟨.result 25088 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25088 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨131⟩⟩) (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25087.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25087.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound186019

namespace LeftBound186024
def owner : Owner := ⟨.program ⟨257⟩, ⟨18352⟩⟩
def transferEvent : Nat := 186024
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 186022 .coefficient) (.predecessor 1 186023 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 186022 .coefficient)
      LeftBound186018.bound (LeftBound186018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events726.exact186021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound186018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound186018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 186023 .coefficient)
      LeftAuthority8692.bound (LeftAuthority8692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound186018.bound LeftAuthority8692.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186018.bound, LeftAuthority8692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound186018.actual selector witness) * (LeftAuthority8692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186024

namespace LeftBound186025
def owner : Owner := ⟨.program ⟨257⟩, ⟨18352⟩⟩
def transferEvent : Nat := 186025
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12726⟩⟩], []⟩ [⟨.result 8693 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 8693 .coefficient)
      LeftAuthority8692.bound (LeftAuthority8692.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12726⟩⟩) (rawTerms := some (Proof.Events033.exact8693RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8692.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8692.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority8692.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound186025

namespace LeftBound186026
def owner : Owner := ⟨.program ⟨257⟩, ⟨18352⟩⟩
def transferEvent : Nat := 186026
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 186021 .summary) (.transfer 186025) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 186021 .summary)
      LeftBound186019.bound (LeftBound186019.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18351⟩⟩) (rawTerms := some (Proof.Events726.exact186021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound186019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 186025)
      LeftBound186025.bound (LeftBound186025.actual selector witness) := by
  exact .transfer (LeftBound186025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound186019.bound LeftBound186025.bound
def bound : CoeffClass := .finite ⟨2555904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound186019.bound, LeftBound186025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound186019.actual selector witness) * (LeftBound186025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound186026

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
