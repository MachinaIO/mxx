import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1361
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1363
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1364
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1365
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1367
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1368
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1369
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1371
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1372
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1385

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound207092
def owner : Owner := ⟨.program ⟨257⟩, ⟨70325⟩⟩
def transferEvent : Nat := 207092
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207088 .summary, .result 205108 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207088 .summary)
      LeftBound207087.bound (LeftBound207087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64932⟩⟩) (rawTerms := some (Proof.Events808.exact207088RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 205108 .summary)
      LeftBound205103.bound (LeftBound205103.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70324⟩⟩) (rawTerms := some (Proof.Events801.exact205108RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound205103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207087.bound, LeftBound205103.bound]
def bound : CoeffClass := .finite ⟨3456353380086899479155517117627148481331252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207087.bound, LeftBound205103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207087.actual selector witness, LeftBound205103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207092

namespace LeftBound207096
def owner : Owner := ⟨.program ⟨257⟩, ⟨70326⟩⟩
def transferEvent : Nat := 207096
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207094 .coefficient, .predecessor 1 207095 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207094 .coefficient)
      LeftBound207091.bound (LeftBound207091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207095 .coefficient)
      LeftBound204889.bound (LeftBound204889.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events800.exact204896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204889.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204889.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207091.bound, LeftBound204889.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207091.bound, LeftBound204889.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207091.actual selector witness, LeftBound204889.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207096

namespace LeftBound207097
def owner : Owner := ⟨.program ⟨257⟩, ⟨70326⟩⟩
def transferEvent : Nat := 207097
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207093 .summary, .result 204896 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207093 .summary)
      LeftBound207092.bound (LeftBound207092.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70325⟩⟩) (rawTerms := some (Proof.Events808.exact207093RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204896 .summary)
      LeftBound204891.bound (LeftBound204891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28337⟩⟩) (rawTerms := some (Proof.Events800.exact204896RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207092.bound, LeftBound204891.bound]
def bound : CoeffClass := .finite ⟨3802007596962448506045899439491360353157172, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207092.bound, LeftBound204891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207092.actual selector witness, LeftBound204891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207097

namespace LeftBound207101
def owner : Owner := ⟨.program ⟨257⟩, ⟨70327⟩⟩
def transferEvent : Nat := 207101
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207099 .coefficient, .predecessor 1 207100 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207099 .coefficient)
      LeftBound207096.bound (LeftBound207096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207096.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207096.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207100 .coefficient)
      LeftBound204677.bound (LeftBound204677.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events799.exact204684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204677.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204677.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207096.bound, LeftBound204677.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207096.bound, LeftBound204677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207096.actual selector witness, LeftBound204677.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207101

namespace LeftBound207102
def owner : Owner := ⟨.program ⟨257⟩, ⟨70327⟩⟩
def transferEvent : Nat := 207102
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207098 .summary, .result 204684 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207098 .summary)
      LeftBound207097.bound (LeftBound207097.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70326⟩⟩) (rawTerms := some (Proof.Events808.exact207098RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204684 .summary)
      LeftBound204679.bound (LeftBound204679.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31017⟩⟩) (rawTerms := some (Proof.Events799.exact204684RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207097.bound, LeftBound204679.bound]
def bound : CoeffClass := .finite ⟨4147668141949793872257454032897973461975092, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207097.bound, LeftBound204679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207097.actual selector witness, LeftBound204679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207102

namespace LeftBound207106
def owner : Owner := ⟨.program ⟨257⟩, ⟨70328⟩⟩
def transferEvent : Nat := 207106
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207104 .coefficient, .predecessor 1 207105 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207104 .coefficient)
      LeftBound207101.bound (LeftBound207101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events808.exact207103RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207105 .coefficient)
      LeftBound204465.bound (LeftBound204465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events798.exact204472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207101.bound, LeftBound204465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207101.bound, LeftBound204465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207101.actual selector witness, LeftBound204465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207106

namespace LeftBound207107
def owner : Owner := ⟨.program ⟨257⟩, ⟨70328⟩⟩
def transferEvent : Nat := 207107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207103 .summary, .result 204472 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207103 .summary)
      LeftBound207102.bound (LeftBound207102.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70327⟩⟩) (rawTerms := some (Proof.Events808.exact207103RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204472 .summary)
      LeftBound204467.bound (LeftBound204467.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36677⟩⟩) (rawTerms := some (Proof.Events798.exact204472RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207102.bound, LeftBound204467.bound]
def bound : CoeffClass := .finite ⟨4493332905678336798016456807332854062121012, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207102.bound, LeftBound204467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207102.actual selector witness, LeftBound204467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207107

namespace LeftBound207111
def owner : Owner := ⟨.program ⟨257⟩, ⟨70329⟩⟩
def transferEvent : Nat := 207111
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207109 .coefficient, .predecessor 1 207110 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207109 .coefficient)
      LeftBound207106.bound (LeftBound207106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207106.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207110 .coefficient)
      LeftBound204253.bound (LeftBound204253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207106.bound, LeftBound204253.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207106.bound, LeftBound204253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207106.actual selector witness, LeftBound204253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207111

namespace LeftBound207112
def owner : Owner := ⟨.program ⟨257⟩, ⟨70329⟩⟩
def transferEvent : Nat := 207112
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207108 .summary, .result 204260 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207108 .summary)
      LeftBound207107.bound (LeftBound207107.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70328⟩⟩) (rawTerms := some (Proof.Events809.exact207108RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204260 .summary)
      LeftBound204255.bound (LeftBound204255.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39357⟩⟩) (rawTerms := some (Proof.Events797.exact204260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204255.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207107.bound, LeftBound204255.bound]
def bound : CoeffClass := .finite ⟨4838999778777478503549183672281868407930932, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207107.bound, LeftBound204255.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207107.actual selector witness, LeftBound204255.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207112

namespace LeftBound207116
def owner : Owner := ⟨.program ⟨257⟩, ⟨70330⟩⟩
def transferEvent : Nat := 207116
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207114 .coefficient, .predecessor 1 207115 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207114 .coefficient)
      LeftBound207111.bound (LeftBound207111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207111.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207115 .coefficient)
      LeftBound204041.bound (LeftBound204041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204041.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207111.bound, LeftBound204041.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207111.bound, LeftBound204041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207111.actual selector witness, LeftBound204041.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207116

namespace LeftBound207117
def owner : Owner := ⟨.program ⟨257⟩, ⟨70330⟩⟩
def transferEvent : Nat := 207117
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207113 .summary, .result 204048 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207113 .summary)
      LeftBound207112.bound (LeftBound207112.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70329⟩⟩) (rawTerms := some (Proof.Events809.exact207113RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204048 .summary)
      LeftBound204043.bound (LeftBound204043.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨42037⟩⟩) (rawTerms := some (Proof.Events797.exact204048RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204043.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207112.bound, LeftBound204043.bound]
def bound : CoeffClass := .finite ⟨5184670870617817768629358718259150245068852, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207112.bound, LeftBound204043.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207112.actual selector witness, LeftBound204043.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207117

namespace LeftBound207121
def owner : Owner := ⟨.program ⟨257⟩, ⟨70331⟩⟩
def transferEvent : Nat := 207121
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207119 .coefficient, .predecessor 1 207120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207119 .coefficient)
      LeftBound207116.bound (LeftBound207116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207116.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207120 .coefficient)
      LeftBound203829.bound (LeftBound203829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events796.exact203836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203829.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207116.bound, LeftBound203829.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207116.bound, LeftBound203829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207116.actual selector witness, LeftBound203829.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207121

namespace LeftBound207122
def owner : Owner := ⟨.program ⟨257⟩, ⟨70331⟩⟩
def transferEvent : Nat := 207122
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207118 .summary, .result 203836 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207118 .summary)
      LeftBound207117.bound (LeftBound207117.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70330⟩⟩) (rawTerms := some (Proof.Events809.exact207118RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 203836 .summary)
      LeftBound203831.bound (LeftBound203831.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44717⟩⟩) (rawTerms := some (Proof.Events796.exact203836RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound203831.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207117.bound, LeftBound203831.bound]
def bound : CoeffClass := .finite ⟨5530348290569953373030706035778833319198772, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207117.bound, LeftBound203831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207117.actual selector witness, LeftBound203831.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207122

namespace LeftBound207126
def owner : Owner := ⟨.program ⟨257⟩, ⟨70332⟩⟩
def transferEvent : Nat := 207126
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207124 .coefficient, .predecessor 1 207125 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207124 .coefficient)
      LeftBound207121.bound (LeftBound207121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207123RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207125 .coefficient)
      LeftBound203617.bound (LeftBound203617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events795.exact203624RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203617.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207121.bound, LeftBound203617.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207121.bound, LeftBound203617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207121.actual selector witness, LeftBound203617.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207126

namespace LeftBound207127
def owner : Owner := ⟨.program ⟨257⟩, ⟨70332⟩⟩
def transferEvent : Nat := 207127
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 207123 .summary, .result 203624 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207123 .summary)
      LeftBound207122.bound (LeftBound207122.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70331⟩⟩) (rawTerms := some (Proof.Events809.exact207123RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207122.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 203624 .summary)
      LeftBound203619.bound (LeftBound203619.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47397⟩⟩) (rawTerms := some (Proof.Events795.exact203624RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound203619.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207122.bound, LeftBound203619.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207122.bound, LeftBound203619.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207122.actual selector witness, LeftBound203619.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207127

namespace LeftBound207131
def owner : Owner := ⟨.program ⟨257⟩, ⟨70333⟩⟩
def transferEvent : Nat := 207131
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 207129 .coefficient, .predecessor 1 207130 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 207129 .coefficient)
      LeftBound207126.bound (LeftBound207126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events809.exact207128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 207130 .coefficient)
      LeftBound203405.bound (LeftBound203405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events794.exact203412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound203405.bound, RecordedBoundRefines] <;> decide)
      (LeftBound203405.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound207126.bound, LeftBound203405.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207126.bound, LeftBound203405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound207126.actual selector witness, LeftBound203405.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound207131

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
