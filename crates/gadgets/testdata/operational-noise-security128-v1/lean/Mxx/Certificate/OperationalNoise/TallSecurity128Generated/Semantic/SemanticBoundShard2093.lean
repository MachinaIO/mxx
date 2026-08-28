import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2072
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2073
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2074
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2075
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2076
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2077
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2078
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2079
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2080
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2092

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound307967
def owner : Owner := ⟨.program ⟨257⟩, ⟨64560⟩⟩
def transferEvent : Nat := 307967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307963 .summary, .result 306392 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307963 .summary)
      LeftBound307962.bound (LeftBound307962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61580⟩⟩) (rawTerms := some (Proof.Events1202.exact307963RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306392 .summary)
      LeftBound306387.bound (LeftBound306387.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64559⟩⟩) (rawTerms := some (Proof.Events1196.exact306392RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307962.bound, LeftBound306387.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307962.bound, LeftBound306387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307962.actual selector witness, LeftBound306387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307967

namespace LeftBound307971
def owner : Owner := ⟨.program ⟨257⟩, ⟨69377⟩⟩
def transferEvent : Nat := 307971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307969 .coefficient, .predecessor 1 307970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307969 .coefficient)
      LeftBound307966.bound (LeftBound307966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307970 .coefficient)
      LeftBound306197.bound (LeftBound306197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1196.exact306204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307966.bound, LeftBound306197.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307966.bound, LeftBound306197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307966.actual selector witness, LeftBound306197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307971

namespace LeftBound307972
def owner : Owner := ⟨.program ⟨257⟩, ⟨69377⟩⟩
def transferEvent : Nat := 307972
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307968 .summary, .result 306204 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307968 .summary)
      LeftBound307967.bound (LeftBound307967.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64560⟩⟩) (rawTerms := some (Proof.Events1203.exact307968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306204 .summary)
      LeftBound306199.bound (LeftBound306199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69376⟩⟩) (rawTerms := some (Proof.Events1196.exact306204RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306199.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307967.bound, LeftBound306199.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307967.bound, LeftBound306199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307967.actual selector witness, LeftBound306199.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307972

namespace LeftBound307976
def owner : Owner := ⟨.program ⟨257⟩, ⟨69378⟩⟩
def transferEvent : Nat := 307976
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307974 .coefficient, .predecessor 1 307975 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307974 .coefficient)
      LeftBound307971.bound (LeftBound307971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307975 .coefficient)
      LeftBound306009.bound (LeftBound306009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1195.exact306016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound306009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound306009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307971.bound, LeftBound306009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307971.bound, LeftBound306009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307971.actual selector witness, LeftBound306009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307976

namespace LeftBound307977
def owner : Owner := ⟨.program ⟨257⟩, ⟨69378⟩⟩
def transferEvent : Nat := 307977
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307973 .summary, .result 306016 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307973 .summary)
      LeftBound307972.bound (LeftBound307972.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69377⟩⟩) (rawTerms := some (Proof.Events1203.exact307973RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 306016 .summary)
      LeftBound306011.bound (LeftBound306011.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28037⟩⟩) (rawTerms := some (Proof.Events1195.exact306016RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound306011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307972.bound, LeftBound306011.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307972.bound, LeftBound306011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307972.actual selector witness, LeftBound306011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307977

namespace LeftBound307981
def owner : Owner := ⟨.program ⟨257⟩, ⟨69379⟩⟩
def transferEvent : Nat := 307981
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307979 .coefficient, .predecessor 1 307980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307979 .coefficient)
      LeftBound307976.bound (LeftBound307976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307976.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307980 .coefficient)
      LeftBound305821.bound (LeftBound305821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1194.exact305828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305821.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307976.bound, LeftBound305821.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307976.bound, LeftBound305821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307976.actual selector witness, LeftBound305821.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307981

namespace LeftBound307982
def owner : Owner := ⟨.program ⟨257⟩, ⟨69379⟩⟩
def transferEvent : Nat := 307982
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307978 .summary, .result 305828 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307978 .summary)
      LeftBound307977.bound (LeftBound307977.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69378⟩⟩) (rawTerms := some (Proof.Events1203.exact307978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305828 .summary)
      LeftBound305823.bound (LeftBound305823.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30717⟩⟩) (rawTerms := some (Proof.Events1194.exact305828RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305823.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307977.bound, LeftBound305823.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307977.bound, LeftBound305823.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307977.actual selector witness, LeftBound305823.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307982

namespace LeftBound307986
def owner : Owner := ⟨.program ⟨257⟩, ⟨69380⟩⟩
def transferEvent : Nat := 307986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307984 .coefficient, .predecessor 1 307985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307984 .coefficient)
      LeftBound307981.bound (LeftBound307981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307985 .coefficient)
      LeftBound305633.bound (LeftBound305633.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1193.exact305640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305633.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307981.bound, LeftBound305633.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307981.bound, LeftBound305633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307981.actual selector witness, LeftBound305633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307986

namespace LeftBound307987
def owner : Owner := ⟨.program ⟨257⟩, ⟨69380⟩⟩
def transferEvent : Nat := 307987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307983 .summary, .result 305640 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307983 .summary)
      LeftBound307982.bound (LeftBound307982.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69379⟩⟩) (rawTerms := some (Proof.Events1203.exact307983RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305640 .summary)
      LeftBound305635.bound (LeftBound305635.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36377⟩⟩) (rawTerms := some (Proof.Events1193.exact305640RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305635.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307982.bound, LeftBound305635.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307982.bound, LeftBound305635.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307982.actual selector witness, LeftBound305635.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307987

namespace LeftBound307991
def owner : Owner := ⟨.program ⟨257⟩, ⟨69381⟩⟩
def transferEvent : Nat := 307991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307989 .coefficient, .predecessor 1 307990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307989 .coefficient)
      LeftBound307986.bound (LeftBound307986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307990 .coefficient)
      LeftBound305445.bound (LeftBound305445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1193.exact305452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307986.bound, LeftBound305445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307986.bound, LeftBound305445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307986.actual selector witness, LeftBound305445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307991

namespace LeftBound307992
def owner : Owner := ⟨.program ⟨257⟩, ⟨69381⟩⟩
def transferEvent : Nat := 307992
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307988 .summary, .result 305452 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307988 .summary)
      LeftBound307987.bound (LeftBound307987.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69380⟩⟩) (rawTerms := some (Proof.Events1203.exact307988RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305452 .summary)
      LeftBound305447.bound (LeftBound305447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39057⟩⟩) (rawTerms := some (Proof.Events1193.exact305452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307987.bound, LeftBound305447.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307987.bound, LeftBound305447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307987.actual selector witness, LeftBound305447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307992

namespace LeftBound307996
def owner : Owner := ⟨.program ⟨257⟩, ⟨69382⟩⟩
def transferEvent : Nat := 307996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307994 .coefficient, .predecessor 1 307995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307994 .coefficient)
      LeftBound307991.bound (LeftBound307991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 307995 .coefficient)
      LeftBound305257.bound (LeftBound305257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1192.exact305264RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307991.bound, LeftBound305257.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307991.bound, LeftBound305257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307991.actual selector witness, LeftBound305257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307996

namespace LeftBound307997
def owner : Owner := ⟨.program ⟨257⟩, ⟨69382⟩⟩
def transferEvent : Nat := 307997
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307993 .summary, .result 305264 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307993 .summary)
      LeftBound307992.bound (LeftBound307992.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69381⟩⟩) (rawTerms := some (Proof.Events1203.exact307993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305264 .summary)
      LeftBound305259.bound (LeftBound305259.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41737⟩⟩) (rawTerms := some (Proof.Events1192.exact305264RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305259.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307992.bound, LeftBound305259.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307992.bound, LeftBound305259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307992.actual selector witness, LeftBound305259.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound307997

namespace LeftBound308001
def owner : Owner := ⟨.program ⟨257⟩, ⟨69383⟩⟩
def transferEvent : Nat := 308001
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 307999 .coefficient, .predecessor 1 308000 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 307999 .coefficient)
      LeftBound307996.bound (LeftBound307996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact307998RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308000 .coefficient)
      LeftBound305069.bound (LeftBound305069.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1191.exact305076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound305069.bound, RecordedBoundRefines] <;> decide)
      (LeftBound305069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307996.bound, LeftBound305069.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307996.bound, LeftBound305069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307996.actual selector witness, LeftBound305069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308001

namespace LeftBound308002
def owner : Owner := ⟨.program ⟨257⟩, ⟨69383⟩⟩
def transferEvent : Nat := 308002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307998 .summary, .result 305076 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307998 .summary)
      LeftBound307997.bound (LeftBound307997.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69382⟩⟩) (rawTerms := some (Proof.Events1203.exact307998RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307997.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 305076 .summary)
      LeftBound305071.bound (LeftBound305071.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44417⟩⟩) (rawTerms := some (Proof.Events1191.exact305076RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound305071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307997.bound, LeftBound305071.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307997.bound, LeftBound305071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307997.actual selector witness, LeftBound305071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308002

namespace LeftBound308006
def owner : Owner := ⟨.program ⟨257⟩, ⟨69384⟩⟩
def transferEvent : Nat := 308006
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308004 .coefficient, .predecessor 1 308005 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308004 .coefficient)
      LeftBound308001.bound (LeftBound308001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308005 .coefficient)
      LeftBound304881.bound (LeftBound304881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1190.exact304888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308001.bound, LeftBound304881.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308001.bound, LeftBound304881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308001.actual selector witness, LeftBound304881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308006

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
