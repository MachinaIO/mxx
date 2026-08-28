import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1207

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound180783
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def transferEvent : Nat := 180783
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 180781 .coefficient) (.predecessor 1 180782 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180781 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180782 .coefficient)
      LeftBound180779.bound (LeftBound180779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180779.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound180779.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound180779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound180779.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180783

namespace LeftBound180784
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def transferEvent : Nat := 180784
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35219⟩⟩]⟩ [⟨.result 180776 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 180776 .coefficient)
      LeftAuthority180775.bound (LeftAuthority180775.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35219⟩⟩) (rawTerms := some (Proof.Events706.exact180776RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180775.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority180775.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority180775.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound180784

namespace LeftBound180785
def owner : Owner := ⟨.program ⟨257⟩, ⟨35222⟩⟩
def transferEvent : Nat := 180785
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 180784) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 180784)
      LeftBound180784.bound (LeftBound180784.actual selector witness) := by
  exact .transfer (LeftBound180784.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound180784.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound180784.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound180784.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180785

namespace LeftBound180864
def owner : Owner := ⟨.program ⟨257⟩, ⟨34507⟩⟩
def transferEvent : Nat := 180864
def frameStart : Nat := 180835
def rule : BoundRule := .product (.predecessor 0 180862 .coefficient) (.predecessor 1 180863 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180862 .coefficient)
      LeftAuthority180860.bound (LeftAuthority180860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180860.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180863 .coefficient)
      LeftAuthority180857.bound (LeftAuthority180857.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180857.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180857.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority180860.bound LeftAuthority180857.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180860.bound, LeftAuthority180857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority180860.actual selector witness) * (LeftAuthority180857.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180864

namespace LeftBound180868
def owner : Owner := ⟨.program ⟨257⟩, ⟨34508⟩⟩
def transferEvent : Nat := 180868
def frameStart : Nat := 180835
def rule : BoundRule := .identity (.predecessor 0 180867 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180867 .coefficient)
      LeftBound180864.bound (LeftBound180864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180866RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180864.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180864.derived selector witness)

def rawBound : CoeffClass := LeftBound180864.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound180864.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound180868

namespace LeftBound180885
def owner : Owner := ⟨.program ⟨257⟩, ⟨36038⟩⟩
def transferEvent : Nat := 180885
def frameStart : Nat := 180835
def rule : BoundRule := .sum [.predecessor 0 180883 .coefficient, .predecessor 1 180884 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180883 .coefficient)
      LeftBound180868.bound (LeftBound180868.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound180868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180884 .coefficient)
      LeftAuthority180881.bound (LeftAuthority180881.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority180881.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound180868.bound, LeftAuthority180881.bound]
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180868.bound, LeftAuthority180881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound180868.actual selector witness, LeftAuthority180881.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound180885

namespace LeftBound180888
def owner : Owner := ⟨.program ⟨257⟩, ⟨36039⟩⟩
def transferEvent : Nat := 180888
def frameStart : Nat := 180835
def rule : BoundRule := .identity (.predecessor 0 180887 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180887 .coefficient)
      LeftBound180885.bound (LeftBound180885.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound180885.derived selector witness)

def rawBound : CoeffClass := LeftBound180885.bound
def bound : CoeffClass := .finite ⟨1600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound180885.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound180888

namespace LeftBound180894
def owner : Owner := ⟨.program ⟨257⟩, ⟨36040⟩⟩
def transferEvent : Nat := 180894
def frameStart : Nat := 180835
def rule : BoundRule := .product (.predecessor 0 180892 .coefficient) (.predecessor 1 180893 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180892 .coefficient)
      LeftAuthority180890.bound (LeftAuthority180890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180893 .coefficient)
      LeftBound180888.bound (LeftBound180888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180888.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority180890.bound LeftBound180888.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180890.bound, LeftBound180888.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority180890.actual selector witness) * (LeftBound180888.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180894

namespace LeftBound180910
def owner : Owner := ⟨.program ⟨257⟩, ⟨9551⟩⟩
def transferEvent : Nat := 180910
def frameStart : Nat := 180835
def rule : BoundRule := .scale (.predecessor 0 180908 .coefficient) (.value (.predecessor 1 180909 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180908 .coefficient)
      LeftAuthority180906.bound (LeftAuthority180906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180909 .coefficient)
      LeftAuthority180897.bound (LeftAuthority180897.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority180897.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority180906.bound LeftAuthority180897.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180906.bound, LeftAuthority180897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority180906.actual selector witness) * (LeftAuthority180897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound180910

namespace LeftBound180913
def owner : Owner := ⟨.program ⟨257⟩, ⟨7297⟩⟩
def transferEvent : Nat := 180913
def frameStart : Nat := 180835
def rule : BoundRule := .identity (.predecessor 0 180912 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180912 .coefficient)
      LeftAuthority180900.bound (LeftAuthority180900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180900.derived selector witness)

def rawBound : CoeffClass := LeftAuthority180900.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority180900.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound180913

namespace LeftBound180917
def owner : Owner := ⟨.program ⟨257⟩, ⟨9552⟩⟩
def transferEvent : Nat := 180917
def frameStart : Nat := 180835
def rule : BoundRule := .product (.predecessor 0 180915 .coefficient) (.predecessor 1 180916 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180915 .coefficient)
      LeftBound180913.bound (LeftBound180913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180916 .coefficient)
      LeftBound180910.bound (LeftBound180910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180910.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound180913.bound LeftBound180910.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180913.bound, LeftBound180910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound180913.actual selector witness) * (LeftBound180910.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180917

namespace LeftBound180922
def owner : Owner := ⟨.program ⟨257⟩, ⟨36041⟩⟩
def transferEvent : Nat := 180922
def frameStart : Nat := 180835
def rule : BoundRule := .sum [.predecessor 0 180920 .coefficient, .predecessor 1 180921 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180920 .coefficient)
      LeftBound180917.bound (LeftBound180917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180921 .coefficient)
      LeftBound180894.bound (LeftBound180894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180894.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180894.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound180917.bound, LeftBound180894.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180917.bound, LeftBound180894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound180917.actual selector witness, LeftBound180894.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound180922

namespace LeftBound180926
def owner : Owner := ⟨.program ⟨257⟩, ⟨36295⟩⟩
def transferEvent : Nat := 180926
def frameStart : Nat := 180835
def rule : BoundRule := .product (.predecessor 0 180924 .coefficient) (.predecessor 1 180925 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180924 .coefficient)
      LeftBound180922.bound (LeftBound180922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180925 .coefficient)
      LeftAuthority180879.bound (LeftAuthority180879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180879.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound180922.bound LeftAuthority180879.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180922.bound, LeftAuthority180879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound180922.actual selector witness) * (LeftAuthority180879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180926

namespace LeftBound180937
def owner : Owner := ⟨.program ⟨257⟩, ⟨34774⟩⟩
def transferEvent : Nat := 180937
def frameStart : Nat := 180835
def rule : BoundRule := .product (.predecessor 0 180935 .coefficient) (.predecessor 1 180936 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180935 .coefficient)
      LeftAuthority180890.bound (LeftAuthority180890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180936 .coefficient)
      LeftAuthority180933.bound (LeftAuthority180933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority180890.bound LeftAuthority180933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180890.bound, LeftAuthority180933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority180890.actual selector witness) * (LeftAuthority180933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound180937

namespace LeftBound180945
def owner : Owner := ⟨.program ⟨257⟩, ⟨34775⟩⟩
def transferEvent : Nat := 180945
def frameStart : Nat := 180835
def rule : BoundRule := .sum [.predecessor 0 180943 .coefficient, .predecessor 1 180944 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180943 .coefficient)
      LeftAuthority180941.bound (LeftAuthority180941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority180941.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority180941.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180944 .coefficient)
      LeftBound180937.bound (LeftBound180937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180937.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority180941.bound, LeftBound180937.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority180941.bound, LeftBound180937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority180941.actual selector witness, LeftBound180937.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound180945

namespace LeftBound180949
def owner : Owner := ⟨.program ⟨257⟩, ⟨36296⟩⟩
def transferEvent : Nat := 180949
def frameStart : Nat := 180835
def rule : BoundRule := .sum [.predecessor 0 180947 .coefficient, .predecessor 1 180948 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 180947 .coefficient)
      LeftBound180945.bound (LeftBound180945.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180946RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180945.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180945.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 180948 .coefficient)
      LeftBound180926.bound (LeftBound180926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events706.exact180931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound180926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound180926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound180945.bound, LeftBound180926.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound180945.bound, LeftBound180926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound180945.actual selector witness, LeftBound180926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound180949

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
