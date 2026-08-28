import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard248
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard250
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard251
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard252
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard253
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard254
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard255
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard256
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard257
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard258
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard269

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46211
def owner : Owner := ⟨.program ⟨257⟩, ⟨65149⟩⟩
def transferEvent : Nat := 46211
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46209 .coefficient, .predecessor 1 46210 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46209 .coefficient)
      LeftBound46206.bound (LeftBound46206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46208RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46210 .coefficient)
      LeftBound44438.bound (LeftBound44438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46206.bound, LeftBound44438.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46206.bound, LeftBound44438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46206.actual selector witness, LeftBound44438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46211

namespace LeftBound46212
def owner : Owner := ⟨.program ⟨257⟩, ⟨65149⟩⟩
def transferEvent : Nat := 46212
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46208 .summary, .result 44445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46208 .summary)
      LeftBound46207.bound (LeftBound46207.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨62169⟩⟩) (rawTerms := some (Proof.Events180.exact46208RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44445 .summary)
      LeftBound44440.bound (LeftBound44440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65148⟩⟩) (rawTerms := some (Proof.Events173.exact44445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46207.bound, LeftBound44440.bound]
def bound : CoeffClass := .finite ⟨3110701272581949232038858886277070355169332, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46207.bound, LeftBound44440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46207.actual selector witness, LeftBound44440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46212

namespace LeftBound46216
def owner : Owner := ⟨.program ⟨257⟩, ⟨70878⟩⟩
def transferEvent : Nat := 46216
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46214 .coefficient, .predecessor 1 46215 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46214 .coefficient)
      LeftBound46211.bound (LeftBound46211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46213RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46211.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46215 .coefficient)
      LeftBound44226.bound (LeftBound44226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44226.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46211.bound, LeftBound44226.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46211.bound, LeftBound44226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46211.actual selector witness, LeftBound44226.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46216

namespace LeftBound46217
def owner : Owner := ⟨.program ⟨257⟩, ⟨70878⟩⟩
def transferEvent : Nat := 46217
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46213 .summary, .result 44233 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46213 .summary)
      LeftBound46212.bound (LeftBound46212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65149⟩⟩) (rawTerms := some (Proof.Events180.exact46213RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46212.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44233 .summary)
      LeftBound44228.bound (LeftBound44228.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70877⟩⟩) (rawTerms := some (Proof.Events172.exact44233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44228.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46212.bound, LeftBound44228.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46212.bound, LeftBound44228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46212.actual selector witness, LeftBound44228.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46217

namespace LeftBound46221
def owner : Owner := ⟨.program ⟨257⟩, ⟨70879⟩⟩
def transferEvent : Nat := 46221
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46219 .coefficient, .predecessor 1 46220 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46219 .coefficient)
      LeftBound46216.bound (LeftBound46216.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46216.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46216.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46220 .coefficient)
      LeftBound44014.bound (LeftBound44014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact44021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46216.bound, LeftBound44014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46216.bound, LeftBound44014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46216.actual selector witness, LeftBound44014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46221

namespace LeftBound46222
def owner : Owner := ⟨.program ⟨257⟩, ⟨70879⟩⟩
def transferEvent : Nat := 46222
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46218 .summary, .result 44021 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46218 .summary)
      LeftBound46217.bound (LeftBound46217.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70878⟩⟩) (rawTerms := some (Proof.Events180.exact46218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46217.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 44021 .summary)
      LeftBound44016.bound (LeftBound44016.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28512⟩⟩) (rawTerms := some (Proof.Events171.exact44021RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44016.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46217.bound, LeftBound44016.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46217.bound, LeftBound44016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46217.actual selector witness, LeftBound44016.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46222

namespace LeftBound46226
def owner : Owner := ⟨.program ⟨257⟩, ⟨70880⟩⟩
def transferEvent : Nat := 46226
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46224 .coefficient, .predecessor 1 46225 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46224 .coefficient)
      LeftBound46221.bound (LeftBound46221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46225 .coefficient)
      LeftBound43802.bound (LeftBound43802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events171.exact43809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46221.bound, LeftBound43802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46221.bound, LeftBound43802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46221.actual selector witness, LeftBound43802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46226

namespace LeftBound46227
def owner : Owner := ⟨.program ⟨257⟩, ⟨70880⟩⟩
def transferEvent : Nat := 46227
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46223 .summary, .result 43809 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46223 .summary)
      LeftBound46222.bound (LeftBound46222.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70879⟩⟩) (rawTerms := some (Proof.Events180.exact46223RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46222.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43809 .summary)
      LeftBound43804.bound (LeftBound43804.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31192⟩⟩) (rawTerms := some (Proof.Events171.exact43809RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46222.bound, LeftBound43804.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46222.bound, LeftBound43804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46222.actual selector witness, LeftBound43804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46227

namespace LeftBound46231
def owner : Owner := ⟨.program ⟨257⟩, ⟨70881⟩⟩
def transferEvent : Nat := 46231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46229 .coefficient, .predecessor 1 46230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46229 .coefficient)
      LeftBound46226.bound (LeftBound46226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46230 .coefficient)
      LeftBound43590.bound (LeftBound43590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events170.exact43597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46226.bound, LeftBound43590.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46226.bound, LeftBound43590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46226.actual selector witness, LeftBound43590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46231

namespace LeftBound46232
def owner : Owner := ⟨.program ⟨257⟩, ⟨70881⟩⟩
def transferEvent : Nat := 46232
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46228 .summary, .result 43597 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46228 .summary)
      LeftBound46227.bound (LeftBound46227.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70880⟩⟩) (rawTerms := some (Proof.Events180.exact46228RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43597 .summary)
      LeftBound43592.bound (LeftBound43592.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36852⟩⟩) (rawTerms := some (Proof.Events170.exact43597RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43592.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46227.bound, LeftBound43592.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46227.bound, LeftBound43592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46227.actual selector witness, LeftBound43592.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46232

namespace LeftBound46236
def owner : Owner := ⟨.program ⟨257⟩, ⟨70882⟩⟩
def transferEvent : Nat := 46236
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46234 .coefficient, .predecessor 1 46235 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46234 .coefficient)
      LeftBound46231.bound (LeftBound46231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46235 .coefficient)
      LeftBound43378.bound (LeftBound43378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events169.exact43385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46231.bound, LeftBound43378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46231.bound, LeftBound43378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46231.actual selector witness, LeftBound43378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46236

namespace LeftBound46237
def owner : Owner := ⟨.program ⟨257⟩, ⟨70882⟩⟩
def transferEvent : Nat := 46237
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46233 .summary, .result 43385 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46233 .summary)
      LeftBound46232.bound (LeftBound46232.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70881⟩⟩) (rawTerms := some (Proof.Events180.exact46233RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43385 .summary)
      LeftBound43380.bound (LeftBound43380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39532⟩⟩) (rawTerms := some (Proof.Events169.exact43385RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43380.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46232.bound, LeftBound43380.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46232.bound, LeftBound43380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46232.actual selector witness, LeftBound43380.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46237

namespace LeftBound46241
def owner : Owner := ⟨.program ⟨257⟩, ⟨70883⟩⟩
def transferEvent : Nat := 46241
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46239 .coefficient, .predecessor 1 46240 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46239 .coefficient)
      LeftBound46236.bound (LeftBound46236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46240 .coefficient)
      LeftBound43166.bound (LeftBound43166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events168.exact43173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound43166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound43166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46236.bound, LeftBound43166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46236.bound, LeftBound43166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46236.actual selector witness, LeftBound43166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46241

namespace LeftBound46242
def owner : Owner := ⟨.program ⟨257⟩, ⟨70883⟩⟩
def transferEvent : Nat := 46242
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46238 .summary, .result 43173 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46238 .summary)
      LeftBound46237.bound (LeftBound46237.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70882⟩⟩) (rawTerms := some (Proof.Events180.exact46238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 43173 .summary)
      LeftBound43168.bound (LeftBound43168.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42212⟩⟩) (rawTerms := some (Proof.Events168.exact43173RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound43168.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46237.bound, LeftBound43168.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46237.bound, LeftBound43168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46237.actual selector witness, LeftBound43168.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46242

namespace LeftBound46246
def owner : Owner := ⟨.program ⟨257⟩, ⟨70884⟩⟩
def transferEvent : Nat := 46246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46244 .coefficient, .predecessor 1 46245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 46244 .coefficient)
      LeftBound46241.bound (LeftBound46241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 46245 .coefficient)
      LeftBound42954.bound (LeftBound42954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events167.exact42961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46241.bound, LeftBound42954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46241.bound, LeftBound42954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46241.actual selector witness, LeftBound42954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46246

namespace LeftBound46247
def owner : Owner := ⟨.program ⟨257⟩, ⟨70884⟩⟩
def transferEvent : Nat := 46247
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46243 .summary, .result 42961 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46243 .summary)
      LeftBound46242.bound (LeftBound46242.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70883⟩⟩) (rawTerms := some (Proof.Events180.exact46243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 42961 .summary)
      LeftBound42956.bound (LeftBound42956.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44892⟩⟩) (rawTerms := some (Proof.Events167.exact42961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42956.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46242.bound, LeftBound42956.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46242.bound, LeftBound42956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound46242.actual selector witness, LeftBound42956.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46247

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
