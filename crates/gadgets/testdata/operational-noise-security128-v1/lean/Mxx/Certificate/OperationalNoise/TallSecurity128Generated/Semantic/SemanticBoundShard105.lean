import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard103
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard104

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound21843
def owner : Owner := ⟨.program ⟨257⟩, ⟨62741⟩⟩
def transferEvent : Nat := 21843
def frameStart : Nat := 21733
def rule : BoundRule := .sum [.predecessor 0 21841 .coefficient, .predecessor 1 21842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21841 .coefficient)
      LeftAuthority21839.bound (LeftAuthority21839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21842 .coefficient)
      LeftBound21835.bound (LeftBound21835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority21839.bound, LeftBound21835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21839.bound, LeftBound21835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority21839.actual selector witness, LeftBound21835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21843

namespace LeftBound21847
def owner : Owner := ⟨.program ⟨257⟩, ⟨64347⟩⟩
def transferEvent : Nat := 21847
def frameStart : Nat := 21733
def rule : BoundRule := .sum [.predecessor 0 21845 .coefficient, .predecessor 1 21846 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21845 .coefficient)
      LeftBound21843.bound (LeftBound21843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21846 .coefficient)
      LeftBound21824.bound (LeftBound21824.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21824.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21824.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21843.bound, LeftBound21824.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21843.bound, LeftBound21824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21843.actual selector witness, LeftBound21824.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21847

namespace LeftBound21860
def owner : Owner := ⟨.program ⟨257⟩, ⟨64345⟩⟩
def transferEvent : Nat := 21860
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 21858 .coefficient, .predecessor 1 21859 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21858 .coefficient)
      LeftBound21681.bound (LeftBound21681.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21681.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21681.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21859 .coefficient)
      LeftBound21664.bound (LeftBound21664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21681.bound, LeftBound21664.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21681.bound, LeftBound21664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21681.actual selector witness, LeftBound21664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21860

namespace LeftBound21863
def owner : Owner := ⟨.program ⟨257⟩, ⟨64345⟩⟩
def transferEvent : Nat := 21863
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 21857 .summary, .result 21671 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21857 .summary)
      LeftBound21683.bound (LeftBound21683.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨63285⟩⟩) (rawTerms := some (Proof.Events085.exact21857RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21671 .summary)
      LeftBound21666.bound (LeftBound21666.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64344⟩⟩) (rawTerms := some (Proof.Events084.exact21671RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21683.bound, LeftBound21666.bound]
def bound : CoeffClass := .finite ⟨2997999239428004118528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21683.bound, LeftBound21666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21683.actual selector witness, LeftBound21666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21863

namespace LeftBound21867
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def transferEvent : Nat := 21867
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21865 .coefficient) (.predecessor 1 21866 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21865 .coefficient)
      LeftBound21860.bound (LeftBound21860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21866 .coefficient)
      LeftAuthority21567.bound (LeftAuthority21567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21860.bound LeftAuthority21567.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21860.bound, LeftAuthority21567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21860.actual selector witness) * (LeftAuthority21567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21867

namespace LeftBound21868
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def transferEvent : Nat := 21868
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨64602⟩⟩]⟩ [⟨.result 21568 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21568 .coefficient)
      LeftAuthority21567.bound (LeftAuthority21567.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨64602⟩⟩) (rawTerms := some (Proof.Events084.exact21568RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21567.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21567.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21567.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21567.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21868

namespace LeftBound21869
def owner : Owner := ⟨.program ⟨257⟩, ⟨64604⟩⟩
def transferEvent : Nat := 21869
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21864 .summary) (.transfer 21868) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21864 .summary)
      LeftBound21863.bound (LeftBound21863.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨64345⟩⟩) (rawTerms := some (Proof.Events085.exact21864RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21863.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 21868)
      LeftBound21868.bound (LeftBound21868.actual selector witness) := by
  exact .transfer (LeftBound21868.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound21863.bound LeftBound21868.bound
def bound : CoeffClass := .finite ⟨32190771716940378589077669150720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21863.bound, LeftBound21868.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound21863.actual selector witness) * (LeftBound21868.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21869

namespace LeftBound21880
def owner : Owner := ⟨.program ⟨257⟩, ⟨63504⟩⟩
def transferEvent : Nat := 21880
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 21878 .coefficient) (.value (.predecessor 1 21879 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21878 .coefficient)
      LeftAuthority21876.bound (LeftAuthority21876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21876.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21879 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority21876.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21876.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21876.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound21880

namespace LeftBound21884
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def transferEvent : Nat := 21884
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 21882 .coefficient) (.predecessor 1 21883 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21882 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21883 .coefficient)
      LeftBound21880.bound (LeftBound21880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21880.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound21880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound21880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound21880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21884

namespace LeftBound21885
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def transferEvent : Nat := 21885
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨63502⟩⟩]⟩ [⟨.result 21877 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21877 .coefficient)
      LeftAuthority21876.bound (LeftAuthority21876.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨63502⟩⟩) (rawTerms := some (Proof.Events085.exact21877RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21876.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21876.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21876.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound21885

namespace LeftBound21886
def owner : Owner := ⟨.program ⟨257⟩, ⟨63505⟩⟩
def transferEvent : Nat := 21886
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 21885) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 21885)
      LeftBound21885.bound (LeftBound21885.actual selector witness) := by
  exact .transfer (LeftBound21885.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound21885.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound21885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound21885.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound21886

namespace LeftBound21981
def owner : Owner := ⟨.program ⟨257⟩, ⟨62739⟩⟩
def transferEvent : Nat := 21981
def frameStart : Nat := 21942
def rule : BoundRule := .identity (.predecessor 0 21980 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21980 .coefficient)
      LeftAuthority21978.bound (LeftAuthority21978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact21979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21978.derived selector witness)

def rawBound : CoeffClass := LeftAuthority21978.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority21978.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound21981

namespace LeftBound21998
def owner : Owner := ⟨.program ⟨257⟩, ⟨64250⟩⟩
def transferEvent : Nat := 21998
def frameStart : Nat := 21942
def rule : BoundRule := .sum [.predecessor 0 21996 .coefficient, .predecessor 1 21997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 21996 .coefficient)
      LeftBound21981.bound (LeftBound21981.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 21997 .coefficient)
      LeftAuthority21994.bound (LeftAuthority21994.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority21994.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound21981.bound, LeftAuthority21994.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21981.bound, LeftAuthority21994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound21981.actual selector witness, LeftAuthority21994.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound21998

namespace LeftBound22001
def owner : Owner := ⟨.program ⟨257⟩, ⟨64251⟩⟩
def transferEvent : Nat := 22001
def frameStart : Nat := 21942
def rule : BoundRule := .identity (.predecessor 0 22000 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22000 .coefficient)
      LeftBound21998.bound (LeftBound21998.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound21998.derived selector witness)

def rawBound : CoeffClass := LeftBound21998.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound21998.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound22001

namespace LeftBound22007
def owner : Owner := ⟨.program ⟨257⟩, ⟨64252⟩⟩
def transferEvent : Nat := 22007
def frameStart : Nat := 21942
def rule : BoundRule := .product (.predecessor 0 22005 .coefficient) (.predecessor 1 22006 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22005 .coefficient)
      LeftAuthority22003.bound (LeftAuthority22003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact22004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22006 .coefficient)
      LeftBound22001.bound (LeftBound22001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact22002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22001.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22001.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority22003.bound LeftBound22001.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22003.bound, LeftBound22001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority22003.actual selector witness) * (LeftBound22001.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22007

namespace LeftBound22015
def owner : Owner := ⟨.program ⟨257⟩, ⟨64253⟩⟩
def transferEvent : Nat := 22015
def frameStart : Nat := 21942
def rule : BoundRule := .sum [.predecessor 0 22013 .coefficient, .predecessor 1 22014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 22013 .coefficient)
      LeftAuthority22011.bound (LeftAuthority22011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact22012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 22014 .coefficient)
      LeftBound22007.bound (LeftBound22007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events085.exact22009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority22011.bound, LeftBound22007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22011.bound, LeftBound22007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority22011.actual selector witness, LeftBound22007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound22015

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
