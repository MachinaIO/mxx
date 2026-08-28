import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard391
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard395
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard398
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard402
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard405
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard409
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard413
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard416
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard442

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69978
def owner : Owner := ⟨.program ⟨257⟩, ⟨62113⟩⟩
def transferEvent : Nat := 69978
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69976 .coefficient, .predecessor 1 69977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69976 .coefficient)
      LeftBound69973.bound (LeftBound69973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69977 .coefficient)
      LeftBound66567.bound (LeftBound66567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66567.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69973.bound, LeftBound66567.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69973.bound, LeftBound66567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69973.actual selector witness, LeftBound66567.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69978

namespace LeftBound69979
def owner : Owner := ⟨.program ⟨257⟩, ⟨62113⟩⟩
def transferEvent : Nat := 69979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69975 .summary, .result 66571 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69975 .summary)
      LeftBound69974.bound (LeftBound69974.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨59133⟩⟩) (rawTerms := some (Proof.Events273.exact69975RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 66571 .summary)
      LeftBound66570.bound (LeftBound66570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62112⟩⟩) (rawTerms := some (Proof.Events260.exact66571RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66570.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69974.bound, LeftBound66570.bound]
def bound : CoeffClass := .finite ⟨257515860087126057990209472036864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69974.bound, LeftBound66570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69974.actual selector witness, LeftBound66570.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69979

namespace LeftBound69983
def owner : Owner := ⟨.program ⟨257⟩, ⟨65093⟩⟩
def transferEvent : Nat := 69983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69981 .coefficient, .predecessor 1 69982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69981 .coefficient)
      LeftBound69978.bound (LeftBound69978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69982 .coefficient)
      LeftBound66085.bound (LeftBound66085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69978.bound, LeftBound66085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69978.bound, LeftBound66085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69978.actual selector witness, LeftBound66085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69983

namespace LeftBound69984
def owner : Owner := ⟨.program ⟨257⟩, ⟨65093⟩⟩
def transferEvent : Nat := 69984
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69980 .summary, .result 66089 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69980 .summary)
      LeftBound69979.bound (LeftBound69979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62113⟩⟩) (rawTerms := some (Proof.Events273.exact69980RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 66089 .summary)
      LeftBound66088.bound (LeftBound66088.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65092⟩⟩) (rawTerms := some (Proof.Events258.exact66089RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66088.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69979.bound, LeftBound66088.bound]
def bound : CoeffClass := .finite ⟨289706631804066638652128995049472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69979.bound, LeftBound66088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69979.actual selector witness, LeftBound66088.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69984

namespace LeftBound69988
def owner : Owner := ⟨.program ⟨257⟩, ⟨70734⟩⟩
def transferEvent : Nat := 69988
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69986 .coefficient, .predecessor 1 69987 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69986 .coefficient)
      LeftBound69983.bound (LeftBound69983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69987 .coefficient)
      LeftBound65603.bound (LeftBound65603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69983.bound, LeftBound65603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69983.bound, LeftBound65603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69983.actual selector witness, LeftBound65603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69988

namespace LeftBound69989
def owner : Owner := ⟨.program ⟨257⟩, ⟨70734⟩⟩
def transferEvent : Nat := 69989
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69985 .summary, .result 65607 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69985 .summary)
      LeftBound69984.bound (LeftBound69984.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65093⟩⟩) (rawTerms := some (Proof.Events273.exact69985RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69984.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65607 .summary)
      LeftBound65606.bound (LeftBound65606.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70733⟩⟩) (rawTerms := some (Proof.Events256.exact65607RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69984.bound, LeftBound65606.bound]
def bound : CoeffClass := .finite ⟨321897992872344281445771187322880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69984.bound, LeftBound65606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69984.actual selector witness, LeftBound65606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69989

namespace LeftBound69993
def owner : Owner := ⟨.program ⟨257⟩, ⟨70735⟩⟩
def transferEvent : Nat := 69993
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69991 .coefficient, .predecessor 1 69992 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69991 .coefficient)
      LeftBound69988.bound (LeftBound69988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69992 .coefficient)
      LeftBound65121.bound (LeftBound65121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69988.bound, LeftBound65121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69988.bound, LeftBound65121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69988.actual selector witness, LeftBound65121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69993

namespace LeftBound69994
def owner : Owner := ⟨.program ⟨257⟩, ⟨70735⟩⟩
def transferEvent : Nat := 69994
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69990 .summary, .result 65125 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69990 .summary)
      LeftBound69989.bound (LeftBound69989.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70734⟩⟩) (rawTerms := some (Proof.Events273.exact69990RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69989.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65125 .summary)
      LeftBound65124.bound (LeftBound65124.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28467⟩⟩) (rawTerms := some (Proof.Events254.exact65125RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65124.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69989.bound, LeftBound65124.bound]
def bound : CoeffClass := .finite ⟨354089550391067611616654269349888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69989.bound, LeftBound65124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69989.actual selector witness, LeftBound65124.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69994

namespace LeftBound69998
def owner : Owner := ⟨.program ⟨257⟩, ⟨70736⟩⟩
def transferEvent : Nat := 69998
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 69996 .coefficient, .predecessor 1 69997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 69996 .coefficient)
      LeftBound69993.bound (LeftBound69993.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact69995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69993.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 69997 .coefficient)
      LeftBound64639.bound (LeftBound64639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64643RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64639.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69993.bound, LeftBound64639.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69993.bound, LeftBound64639.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69993.actual selector witness, LeftBound64639.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69998

namespace LeftBound69999
def owner : Owner := ⟨.program ⟨257⟩, ⟨70736⟩⟩
def transferEvent : Nat := 69999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 69995 .summary, .result 64643 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 69995 .summary)
      LeftBound69994.bound (LeftBound69994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70735⟩⟩) (rawTerms := some (Proof.Events273.exact69995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64643 .summary)
      LeftBound64642.bound (LeftBound64642.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31147⟩⟩) (rawTerms := some (Proof.Events252.exact64643RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64642.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69994.bound, LeftBound64642.bound]
def bound : CoeffClass := .finite ⟨386281697261128003919260020637696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69994.bound, LeftBound64642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69994.actual selector witness, LeftBound64642.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69999

namespace LeftBound70003
def owner : Owner := ⟨.program ⟨257⟩, ⟨70737⟩⟩
def transferEvent : Nat := 70003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70001 .coefficient, .predecessor 1 70002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 70001 .coefficient)
      LeftBound69998.bound (LeftBound69998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 70002 .coefficient)
      LeftBound64157.bound (LeftBound64157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events250.exact64161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64157.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69998.bound, LeftBound64157.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69998.bound, LeftBound64157.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69998.actual selector witness, LeftBound64157.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70003

namespace LeftBound70004
def owner : Owner := ⟨.program ⟨257⟩, ⟨70737⟩⟩
def transferEvent : Nat := 70004
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70000 .summary, .result 64161 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 70000 .summary)
      LeftBound69999.bound (LeftBound69999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70736⟩⟩) (rawTerms := some (Proof.Events273.exact70000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 64161 .summary)
      LeftBound64160.bound (LeftBound64160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36807⟩⟩) (rawTerms := some (Proof.Events250.exact64161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64160.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69999.bound, LeftBound64160.bound]
def bound : CoeffClass := .finite ⟨418474237032079770976347551432704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69999.bound, LeftBound64160.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound69999.actual selector witness, LeftBound64160.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70004

namespace LeftBound70008
def owner : Owner := ⟨.program ⟨257⟩, ⟨70738⟩⟩
def transferEvent : Nat := 70008
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70006 .coefficient, .predecessor 1 70007 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 70006 .coefficient)
      LeftBound70003.bound (LeftBound70003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 70007 .coefficient)
      LeftBound63675.bound (LeftBound63675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70003.bound, LeftBound63675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70003.bound, LeftBound63675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound70003.actual selector witness, LeftBound63675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70008

namespace LeftBound70009
def owner : Owner := ⟨.program ⟨257⟩, ⟨70738⟩⟩
def transferEvent : Nat := 70009
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70005 .summary, .result 63679 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 70005 .summary)
      LeftBound70004.bound (LeftBound70004.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70737⟩⟩) (rawTerms := some (Proof.Events273.exact70005RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63679 .summary)
      LeftBound63678.bound (LeftBound63678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39487⟩⟩) (rawTerms := some (Proof.Events248.exact63679RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63678.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70004.bound, LeftBound63678.bound]
def bound : CoeffClass := .finite ⟨450666973253477225410675971981312, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70004.bound, LeftBound63678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound70004.actual selector witness, LeftBound63678.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70009

namespace LeftBound70013
def owner : Owner := ⟨.program ⟨257⟩, ⟨70739⟩⟩
def transferEvent : Nat := 70013
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70011 .coefficient, .predecessor 1 70012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 70011 .coefficient)
      LeftBound70008.bound (LeftBound70008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 70012 .coefficient)
      LeftBound63193.bound (LeftBound63193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63193.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63193.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70008.bound, LeftBound63193.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70008.bound, LeftBound63193.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound70008.actual selector witness, LeftBound63193.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70013

namespace LeftBound70014
def owner : Owner := ⟨.program ⟨257⟩, ⟨70739⟩⟩
def transferEvent : Nat := 70014
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70010 .summary, .result 63197 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 70010 .summary)
      LeftBound70009.bound (LeftBound70009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70738⟩⟩) (rawTerms := some (Proof.Events273.exact70010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 63197 .summary)
      LeftBound63196.bound (LeftBound63196.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42167⟩⟩) (rawTerms := some (Proof.Events246.exact63197RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63196.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70009.bound, LeftBound63196.bound]
def bound : CoeffClass := .finite ⟨482860102375766054599486172037120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70009.bound, LeftBound63196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound70009.actual selector witness, LeftBound63196.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70014

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
