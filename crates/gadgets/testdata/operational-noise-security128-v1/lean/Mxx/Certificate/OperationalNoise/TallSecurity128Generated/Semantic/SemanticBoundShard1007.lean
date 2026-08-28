import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard091
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1006

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound151871
def owner : Owner := ⟨.program ⟨257⟩, ⟨36555⟩⟩
def transferEvent : Nat := 151871
def frameStart : Nat := 151794
def rule : BoundRule := .product (.predecessor 0 151869 .coefficient) (.predecessor 1 151870 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151869 .coefficient)
      LeftBound151867.bound (LeftBound151867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151870 .coefficient)
      LeftAuthority151844.bound (LeftAuthority151844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound151867.bound LeftAuthority151844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151867.bound, LeftAuthority151844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound151867.actual selector witness) * (LeftAuthority151844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151871

namespace LeftBound151882
def owner : Owner := ⟨.program ⟨257⟩, ⟨34925⟩⟩
def transferEvent : Nat := 151882
def frameStart : Nat := 151794
def rule : BoundRule := .product (.predecessor 0 151880 .coefficient) (.predecessor 1 151881 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151880 .coefficient)
      LeftAuthority151855.bound (LeftAuthority151855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151855.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151881 .coefficient)
      LeftAuthority151878.bound (LeftAuthority151878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority151855.bound LeftAuthority151878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151855.bound, LeftAuthority151878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority151855.actual selector witness) * (LeftAuthority151878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151882

namespace LeftBound151890
def owner : Owner := ⟨.program ⟨257⟩, ⟨34926⟩⟩
def transferEvent : Nat := 151890
def frameStart : Nat := 151794
def rule : BoundRule := .sum [.predecessor 0 151888 .coefficient, .predecessor 1 151889 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151888 .coefficient)
      LeftAuthority151886.bound (LeftAuthority151886.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151887RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority151886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority151886.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151889 .coefficient)
      LeftBound151882.bound (LeftBound151882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority151886.bound, LeftBound151882.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority151886.bound, LeftBound151882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority151886.actual selector witness, LeftBound151882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151890

namespace LeftBound151894
def owner : Owner := ⟨.program ⟨257⟩, ⟨36558⟩⟩
def transferEvent : Nat := 151894
def frameStart : Nat := 151794
def rule : BoundRule := .sum [.predecessor 0 151892 .coefficient, .predecessor 1 151893 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151892 .coefficient)
      LeftBound151890.bound (LeftBound151890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151893 .coefficient)
      LeftBound151871.bound (LeftBound151871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151890.bound, LeftBound151871.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151890.bound, LeftBound151871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151890.actual selector witness, LeftBound151871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151894

namespace LeftBound151907
def owner : Owner := ⟨.program ⟨257⟩, ⟨36557⟩⟩
def transferEvent : Nat := 151907
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151905 .coefficient, .predecessor 1 151906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151905 .coefficient)
      LeftBound151736.bound (LeftBound151736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151736.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151736.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151906 .coefficient)
      LeftBound151719.bound (LeftBound151719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events592.exact151726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151719.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151719.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151736.bound, LeftBound151719.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151736.bound, LeftBound151719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151736.actual selector witness, LeftBound151719.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151907

namespace LeftBound151910
def owner : Owner := ⟨.program ⟨257⟩, ⟨36557⟩⟩
def transferEvent : Nat := 151910
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 151904 .summary, .result 151726 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151904 .summary)
      LeftBound151738.bound (LeftBound151738.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨35439⟩⟩) (rawTerms := some (Proof.Events593.exact151904RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151738.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151726 .summary)
      LeftBound151721.bound (LeftBound151721.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36556⟩⟩) (rawTerms := some (Proof.Events592.exact151726RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151738.bound, LeftBound151721.bound]
def bound : CoeffClass := .finite ⟨32192539770951767057087530795008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151738.bound, LeftBound151721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151738.actual selector witness, LeftBound151721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151910

namespace LeftBound151934
def owner : Owner := ⟨.program ⟨257⟩, ⟨28705⟩⟩
def transferEvent : Nat := 151934
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 151932 .coefficient) (.predecessor 1 151933 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151932 .coefficient)
      LeftAuthority6963.bound (LeftAuthority6963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151933 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority6963.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6963.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority6963.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound151934

namespace LeftBound151939
def owner : Owner := ⟨.program ⟨257⟩, ⟨8243⟩⟩
def transferEvent : Nat := 151939
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 151937 .coefficient) (.predecessor 1 151938 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151937 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151938 .coefficient)
      LeftBound20085.bound (LeftBound20085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20085.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound20085.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound20085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound20085.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151939

namespace LeftBound151944
def owner : Owner := ⟨.program ⟨257⟩, ⟨28706⟩⟩
def transferEvent : Nat := 151944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151942 .coefficient, .predecessor 1 151943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151942 .coefficient)
      LeftBound151939.bound (LeftBound151939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151943 .coefficient)
      LeftBound151934.bound (LeftBound151934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151939.bound, LeftBound151934.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151939.bound, LeftBound151934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151939.actual selector witness, LeftBound151934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151944

namespace LeftBound151948
def owner : Owner := ⟨.program ⟨257⟩, ⟨28707⟩⟩
def transferEvent : Nat := 151948
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 151946 .coefficient, .predecessor 1 151947 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151946 .coefficient)
      LeftBound151944.bound (LeftBound151944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151947 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound151944.bound, LeftBound20077.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151944.bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound151944.actual selector witness, LeftBound20077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound151948

namespace LeftBound151949
def owner : Owner := ⟨.program ⟨257⟩, ⟨28707⟩⟩
def transferEvent : Nat := 151949
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨105⟩⟩]⟩ [⟨.result 20078 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 20078 .coefficient)
      LeftBound20077.bound (LeftBound20077.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨105⟩⟩) (rawTerms := some (Proof.Events078.exact20078RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20077.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20077.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound20077.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound20077.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound151949

namespace LeftBound151954
def owner : Owner := ⟨.program ⟨257⟩, ⟨28708⟩⟩
def transferEvent : Nat := 151954
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 151952 .coefficient) (.predecessor 1 151953 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151952 .coefficient)
      LeftBound151948.bound (LeftBound151948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events593.exact151951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151948.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151953 .coefficient)
      LeftAuthority6966.bound (LeftAuthority6966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6966.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound151948.bound LeftAuthority6966.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151948.bound, LeftAuthority6966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound151948.actual selector witness) * (LeftAuthority6966.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151954

namespace LeftBound151955
def owner : Owner := ⟨.program ⟨257⟩, ⟨28708⟩⟩
def transferEvent : Nat := 151955
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨13236⟩⟩], []⟩ [⟨.result 6967 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 6967 .coefficient)
      LeftAuthority6966.bound (LeftAuthority6966.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨13236⟩⟩) (rawTerms := some (Proof.Events027.exact6967RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6966.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6966.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority6966.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound151955

namespace LeftBound151956
def owner : Owner := ⟨.program ⟨257⟩, ⟨28708⟩⟩
def transferEvent : Nat := 151956
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 151951 .summary) (.transfer 151955) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151951 .summary)
      LeftBound151949.bound (LeftBound151949.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28707⟩⟩) (rawTerms := some (Proof.Events593.exact151951RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 151955)
      LeftBound151955.bound (LeftBound151955.actual selector witness) := by
  exact .transfer (LeftBound151955.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound151949.bound LeftBound151955.bound
def bound : CoeffClass := .finite ⟨30670848, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151949.bound, LeftBound151955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound151949.actual selector witness) * (LeftBound151955.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151956

namespace LeftBound151962
def owner : Owner := ⟨.program ⟨257⟩, ⟨13237⟩⟩
def transferEvent : Nat := 151962
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 151960 .coefficient) (.predecessor 1 151961 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151960 .coefficient)
      LeftAuthority6966.bound (LeftAuthority6966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events027.exact6967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151961 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority6966.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6966.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority6966.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound151962

namespace LeftBound151967
def owner : Owner := ⟨.program ⟨257⟩, ⟨8260⟩⟩
def transferEvent : Nat := 151967
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 151965 .coefficient) (.predecessor 1 151966 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 151965 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 151966 .coefficient)
      LeftBound20126.bound (LeftBound20126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20126.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound20126.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound20126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound20126.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound151967

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
