import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard420
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard424
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard427
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard431
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard434
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard438
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard441

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69924
def owner : Owner := ⟨.program ⟨257⟩, ⟨16149⟩⟩
def transferEvent : Nat := 69924
def frameStart : Nat := 69828
def rule : BoundRule := .sum [.predecessor 0 69922 .coefficient, .predecessor 1 69923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69922 .coefficient)
      LeftAuthority69920.bound (LeftAuthority69920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69923 .coefficient)
      LeftBound69916.bound (LeftBound69916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority69920.bound, LeftBound69916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69920.bound, LeftBound69916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority69920.actual selector witness, LeftBound69916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69924

namespace LeftBound69928
def owner : Owner := ⟨.program ⟨257⟩, ⟨17961⟩⟩
def transferEvent : Nat := 69928
def frameStart : Nat := 69828
def rule : BoundRule := .sum [.predecessor 0 69926 .coefficient, .predecessor 1 69927 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69926 .coefficient)
      LeftBound69924.bound (LeftBound69924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69927 .coefficient)
      LeftBound69905.bound (LeftBound69905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69905.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69924.bound, LeftBound69905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69924.bound, LeftBound69905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69924.actual selector witness, LeftBound69905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69928

namespace LeftBound69941
def owner : Owner := ⟨.program ⟨257⟩, ⟨17960⟩⟩
def transferEvent : Nat := 69941
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69939 .coefficient, .predecessor 1 69940 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69939 .coefficient)
      LeftBound69770.bound (LeftBound69770.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69770.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69770.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69940 .coefficient)
      LeftBound69753.bound (LeftBound69753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69753.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69770.bound, LeftBound69753.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69770.bound, LeftBound69753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69770.actual selector witness, LeftBound69753.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69941

namespace LeftBound69944
def owner : Owner := ⟨.program ⟨257⟩, ⟨17960⟩⟩
def transferEvent : Nat := 69944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69938 .summary, .result 69760 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69938 .summary)
      LeftBound69772.bound (LeftBound69772.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16739⟩⟩) (rawTerms := some (Proof.Events273.exact69938RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69772.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69760 .summary)
      LeftBound69755.bound (LeftBound69755.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17959⟩⟩) (rawTerms := some (Proof.Events272.exact69760RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69755.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69772.bound, LeftBound69755.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69772.bound, LeftBound69755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69772.actual selector witness, LeftBound69755.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69944

namespace LeftBound69948
def owner : Owner := ⟨.program ⟨257⟩, ⟨20873⟩⟩
def transferEvent : Nat := 69948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69946 .coefficient, .predecessor 1 69947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69946 .coefficient)
      LeftBound69941.bound (LeftBound69941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69947 .coefficient)
      LeftBound69459.bound (LeftBound69459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events271.exact69463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69459.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69941.bound, LeftBound69459.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69941.bound, LeftBound69459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69941.actual selector witness, LeftBound69459.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69948

namespace LeftBound69949
def owner : Owner := ⟨.program ⟨257⟩, ⟨20873⟩⟩
def transferEvent : Nat := 69949
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69945 .summary, .result 69463 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69945 .summary)
      LeftBound69944.bound (LeftBound69944.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17960⟩⟩) (rawTerms := some (Proof.Events273.exact69945RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69463 .summary)
      LeftBound69462.bound (LeftBound69462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20872⟩⟩) (rawTerms := some (Proof.Events271.exact69463RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69462.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69944.bound, LeftBound69462.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69944.bound, LeftBound69462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69944.actual selector witness, LeftBound69462.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69949

namespace LeftBound69953
def owner : Owner := ⟨.program ⟨257⟩, ⟨24093⟩⟩
def transferEvent : Nat := 69953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69951 .coefficient, .predecessor 1 69952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69951 .coefficient)
      LeftBound69948.bound (LeftBound69948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69952 .coefficient)
      LeftBound68977.bound (LeftBound68977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events269.exact68981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69948.bound, LeftBound68977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69948.bound, LeftBound68977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69948.actual selector witness, LeftBound68977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69953

namespace LeftBound69954
def owner : Owner := ⟨.program ⟨257⟩, ⟨24093⟩⟩
def transferEvent : Nat := 69954
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69950 .summary, .result 68981 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69950 .summary)
      LeftBound69949.bound (LeftBound69949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20873⟩⟩) (rawTerms := some (Proof.Events273.exact69950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68981 .summary)
      LeftBound68980.bound (LeftBound68980.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24092⟩⟩) (rawTerms := some (Proof.Events269.exact68981RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68980.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69949.bound, LeftBound68980.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69949.bound, LeftBound68980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69949.actual selector witness, LeftBound68980.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69954

namespace LeftBound69958
def owner : Owner := ⟨.program ⟨257⟩, ⟨34113⟩⟩
def transferEvent : Nat := 69958
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69956 .coefficient, .predecessor 1 69957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69956 .coefficient)
      LeftBound69953.bound (LeftBound69953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69957 .coefficient)
      LeftBound68495.bound (LeftBound68495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events267.exact68499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68495.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69953.bound, LeftBound68495.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69953.bound, LeftBound68495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69953.actual selector witness, LeftBound68495.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69958

namespace LeftBound69959
def owner : Owner := ⟨.program ⟨257⟩, ⟨34113⟩⟩
def transferEvent : Nat := 69959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69955 .summary, .result 68499 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69955 .summary)
      LeftBound69954.bound (LeftBound69954.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24093⟩⟩) (rawTerms := some (Proof.Events273.exact69955RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68499 .summary)
      LeftBound68498.bound (LeftBound68498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34112⟩⟩) (rawTerms := some (Proof.Events267.exact68499RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69954.bound, LeftBound68498.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69954.bound, LeftBound68498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69954.actual selector witness, LeftBound68498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69959

namespace LeftBound69963
def owner : Owner := ⟨.program ⟨257⟩, ⟨53173⟩⟩
def transferEvent : Nat := 69963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69961 .coefficient, .predecessor 1 69962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69961 .coefficient)
      LeftBound69958.bound (LeftBound69958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69962 .coefficient)
      LeftBound68013.bound (LeftBound68013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events265.exact68017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound68013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound68013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69958.bound, LeftBound68013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69958.bound, LeftBound68013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69958.actual selector witness, LeftBound68013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69963

namespace LeftBound69964
def owner : Owner := ⟨.program ⟨257⟩, ⟨53173⟩⟩
def transferEvent : Nat := 69964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69960 .summary, .result 68017 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69960 .summary)
      LeftBound69959.bound (LeftBound69959.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨34113⟩⟩) (rawTerms := some (Proof.Events273.exact69960RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 68017 .summary)
      LeftBound68016.bound (LeftBound68016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53172⟩⟩) (rawTerms := some (Proof.Events265.exact68017RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound68016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69959.bound, LeftBound68016.bound]
def bound : CoeffClass := .finite ⟨160945509440761189776859800535040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69959.bound, LeftBound68016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69959.actual selector witness, LeftBound68016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69964

namespace LeftBound69968
def owner : Owner := ⟨.program ⟨257⟩, ⟨56153⟩⟩
def transferEvent : Nat := 69968
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69966 .coefficient, .predecessor 1 69967 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69966 .coefficient)
      LeftBound69963.bound (LeftBound69963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69967 .coefficient)
      LeftBound67531.bound (LeftBound67531.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events263.exact67535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67531.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67531.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69963.bound, LeftBound67531.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69963.bound, LeftBound67531.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69963.actual selector witness, LeftBound67531.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69968

namespace LeftBound69969
def owner : Owner := ⟨.program ⟨257⟩, ⟨56153⟩⟩
def transferEvent : Nat := 69969
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69965 .summary, .result 67535 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69965 .summary)
      LeftBound69964.bound (LeftBound69964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53173⟩⟩) (rawTerms := some (Proof.Events273.exact69965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 67535 .summary)
      LeftBound67534.bound (LeftBound67534.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56152⟩⟩) (rawTerms := some (Proof.Events263.exact67535RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67534.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69964.bound, LeftBound67534.bound]
def bound : CoeffClass := .finite ⟨193135298905473333552574874779648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69964.bound, LeftBound67534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69964.actual selector witness, LeftBound67534.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69969

namespace LeftBound69973
def owner : Owner := ⟨.program ⟨257⟩, ⟨59133⟩⟩
def transferEvent : Nat := 69973
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69971 .coefficient, .predecessor 1 69972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69971 .coefficient)
      LeftBound69968.bound (LeftBound69968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69968.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69968.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69972 .coefficient)
      LeftBound67049.bound (LeftBound67049.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67049.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69968.bound, LeftBound67049.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69968.bound, LeftBound67049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69968.actual selector witness, LeftBound67049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69973

namespace LeftBound69974
def owner : Owner := ⟨.program ⟨257⟩, ⟨59133⟩⟩
def transferEvent : Nat := 69974
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69970 .summary, .result 67053 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69970 .summary)
      LeftBound69969.bound (LeftBound69969.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨56153⟩⟩) (rawTerms := some (Proof.Events273.exact69970RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 67053 .summary)
      LeftBound67052.bound (LeftBound67052.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59132⟩⟩) (rawTerms := some (Proof.Events261.exact67053RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69969.bound, LeftBound67052.bound]
def bound : CoeffClass := .finite ⟨225325481271076852082771728531456, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69969.bound, LeftBound67052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69969.actual selector witness, LeftBound67052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69974

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
