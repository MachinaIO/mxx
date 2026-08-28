import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1670
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1671
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1672
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1674
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1675
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1676
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1678
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1679
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1689

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound250956
def owner : Owner := ⟨.program ⟨257⟩, ⟨61828⟩⟩
def transferEvent : Nat := 250956
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250954 .coefficient, .predecessor 1 250955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250954 .coefficient)
      LeftBound250951.bound (LeftBound250951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250955 .coefficient)
      LeftBound249400.bound (LeftBound249400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events974.exact249407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound249400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound249400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250951.bound, LeftBound249400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250951.bound, LeftBound249400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250951.actual selector witness, LeftBound249400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250956

namespace LeftBound250957
def owner : Owner := ⟨.program ⟨257⟩, ⟨61828⟩⟩
def transferEvent : Nat := 250957
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250953 .summary, .result 249407 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250953 .summary)
      LeftBound250952.bound (LeftBound250952.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨58848⟩⟩) (rawTerms := some (Proof.Events980.exact250953RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 249407 .summary)
      LeftBound249402.bound (LeftBound249402.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61827⟩⟩) (rawTerms := some (Proof.Events974.exact249407RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound249402.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250952.bound, LeftBound249402.bound]
def bound : CoeffClass := .finite ⟨2765055493188795324243372926469393465999412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250952.bound, LeftBound249402.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250952.actual selector witness, LeftBound249402.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250957

namespace LeftBound250961
def owner : Owner := ⟨.program ⟨257⟩, ⟨64808⟩⟩
def transferEvent : Nat := 250961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250959 .coefficient, .predecessor 1 250960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250959 .coefficient)
      LeftBound250956.bound (LeftBound250956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250960 .coefficient)
      LeftBound249188.bound (LeftBound249188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events973.exact249195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound249188.bound, RecordedBoundRefines] <;> decide)
      (LeftBound249188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250956.bound, LeftBound249188.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250956.bound, LeftBound249188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250956.actual selector witness, LeftBound249188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250961

namespace LeftBound250962
def owner : Owner := ⟨.program ⟨257⟩, ⟨64808⟩⟩
def transferEvent : Nat := 250962
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250958 .summary, .result 249195 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250958 .summary)
      LeftBound250957.bound (LeftBound250957.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61828⟩⟩) (rawTerms := some (Proof.Events980.exact250958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250957.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 249195 .summary)
      LeftBound249190.bound (LeftBound249190.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64807⟩⟩) (rawTerms := some (Proof.Events973.exact249195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound249190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250957.bound, LeftBound249190.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250957.bound, LeftBound249190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250957.actual selector witness, LeftBound249190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250962

namespace LeftBound250966
def owner : Owner := ⟨.program ⟨257⟩, ⟨70009⟩⟩
def transferEvent : Nat := 250966
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250964 .coefficient, .predecessor 1 250965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250964 .coefficient)
      LeftBound250961.bound (LeftBound250961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250965 .coefficient)
      LeftBound248976.bound (LeftBound248976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events972.exact248983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound248976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound248976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250961.bound, LeftBound248976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250961.bound, LeftBound248976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250961.actual selector witness, LeftBound248976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250966

namespace LeftBound250967
def owner : Owner := ⟨.program ⟨257⟩, ⟨70009⟩⟩
def transferEvent : Nat := 250967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250963 .summary, .result 248983 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250963 .summary)
      LeftBound250962.bound (LeftBound250962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64808⟩⟩) (rawTerms := some (Proof.Events980.exact250963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 248983 .summary)
      LeftBound248978.bound (LeftBound248978.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70008⟩⟩) (rawTerms := some (Proof.Events972.exact248983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound248978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250962.bound, LeftBound248978.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250962.bound, LeftBound248978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250962.actual selector witness, LeftBound248978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250967

namespace LeftBound250971
def owner : Owner := ⟨.program ⟨257⟩, ⟨70010⟩⟩
def transferEvent : Nat := 250971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250969 .coefficient, .predecessor 1 250970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250969 .coefficient)
      LeftBound250966.bound (LeftBound250966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250970 .coefficient)
      LeftBound248764.bound (LeftBound248764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events971.exact248771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound248764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound248764.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250966.bound, LeftBound248764.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250966.bound, LeftBound248764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250966.actual selector witness, LeftBound248764.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250971

namespace LeftBound250972
def owner : Owner := ⟨.program ⟨257⟩, ⟨70010⟩⟩
def transferEvent : Nat := 250972
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250968 .summary, .result 248771 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250968 .summary)
      LeftBound250967.bound (LeftBound250967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70009⟩⟩) (rawTerms := some (Proof.Events980.exact250968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 248771 .summary)
      LeftBound248766.bound (LeftBound248766.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28237⟩⟩) (rawTerms := some (Proof.Events971.exact248771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound248766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250967.bound, LeftBound248766.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250967.bound, LeftBound248766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250967.actual selector witness, LeftBound248766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250972

namespace LeftBound250976
def owner : Owner := ⟨.program ⟨257⟩, ⟨70011⟩⟩
def transferEvent : Nat := 250976
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250974 .coefficient, .predecessor 1 250975 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250974 .coefficient)
      LeftBound250971.bound (LeftBound250971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250975 .coefficient)
      LeftBound248552.bound (LeftBound248552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events970.exact248559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound248552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound248552.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250971.bound, LeftBound248552.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250971.bound, LeftBound248552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250971.actual selector witness, LeftBound248552.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250976

namespace LeftBound250977
def owner : Owner := ⟨.program ⟨257⟩, ⟨70011⟩⟩
def transferEvent : Nat := 250977
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250973 .summary, .result 248559 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250973 .summary)
      LeftBound250972.bound (LeftBound250972.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70010⟩⟩) (rawTerms := some (Proof.Events980.exact250973RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 248559 .summary)
      LeftBound248554.bound (LeftBound248554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30917⟩⟩) (rawTerms := some (Proof.Events970.exact248559RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound248554.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250972.bound, LeftBound248554.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250972.bound, LeftBound248554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250972.actual selector witness, LeftBound248554.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250977

namespace LeftBound250981
def owner : Owner := ⟨.program ⟨257⟩, ⟨70012⟩⟩
def transferEvent : Nat := 250981
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250979 .coefficient, .predecessor 1 250980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250979 .coefficient)
      LeftBound250976.bound (LeftBound250976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250980 .coefficient)
      LeftBound248340.bound (LeftBound248340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events970.exact248347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound248340.bound, RecordedBoundRefines] <;> decide)
      (LeftBound248340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250976.bound, LeftBound248340.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250976.bound, LeftBound248340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250976.actual selector witness, LeftBound248340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250981

namespace LeftBound250982
def owner : Owner := ⟨.program ⟨257⟩, ⟨70012⟩⟩
def transferEvent : Nat := 250982
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250978 .summary, .result 248347 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250978 .summary)
      LeftBound250977.bound (LeftBound250977.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70011⟩⟩) (rawTerms := some (Proof.Events980.exact250978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 248347 .summary)
      LeftBound248342.bound (LeftBound248342.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36577⟩⟩) (rawTerms := some (Proof.Events970.exact248347RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound248342.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250977.bound, LeftBound248342.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250977.bound, LeftBound248342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250977.actual selector witness, LeftBound248342.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250982

namespace LeftBound250986
def owner : Owner := ⟨.program ⟨257⟩, ⟨70013⟩⟩
def transferEvent : Nat := 250986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250984 .coefficient, .predecessor 1 250985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250984 .coefficient)
      LeftBound250981.bound (LeftBound250981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250985 .coefficient)
      LeftBound248128.bound (LeftBound248128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events969.exact248135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound248128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound248128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250981.bound, LeftBound248128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250981.bound, LeftBound248128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250981.actual selector witness, LeftBound248128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250986

namespace LeftBound250987
def owner : Owner := ⟨.program ⟨257⟩, ⟨70013⟩⟩
def transferEvent : Nat := 250987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250983 .summary, .result 248135 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250983 .summary)
      LeftBound250982.bound (LeftBound250982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70012⟩⟩) (rawTerms := some (Proof.Events980.exact250983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 248135 .summary)
      LeftBound248130.bound (LeftBound248130.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39257⟩⟩) (rawTerms := some (Proof.Events969.exact248135RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound248130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250982.bound, LeftBound248130.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250982.bound, LeftBound248130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250982.actual selector witness, LeftBound248130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250987

namespace LeftBound250991
def owner : Owner := ⟨.program ⟨257⟩, ⟨70014⟩⟩
def transferEvent : Nat := 250991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 250989 .coefficient, .predecessor 1 250990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 250989 .coefficient)
      LeftBound250986.bound (LeftBound250986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events980.exact250988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound250986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound250986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 250990 .coefficient)
      LeftBound247916.bound (LeftBound247916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events968.exact247923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250986.bound, LeftBound247916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250986.bound, LeftBound247916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250986.actual selector witness, LeftBound247916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250991

namespace LeftBound250992
def owner : Owner := ⟨.program ⟨257⟩, ⟨70014⟩⟩
def transferEvent : Nat := 250992
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 250988 .summary, .result 247923 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 250988 .summary)
      LeftBound250987.bound (LeftBound250987.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70013⟩⟩) (rawTerms := some (Proof.Events980.exact250988RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound250987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247923 .summary)
      LeftBound247918.bound (LeftBound247918.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41937⟩⟩) (rawTerms := some (Proof.Events968.exact247923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound247918.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound250987.bound, LeftBound247918.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound250987.bound, LeftBound247918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound250987.actual selector witness, LeftBound247918.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound250992

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
