import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard730

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound111701
def owner : Owner := ⟨.program ⟨257⟩, ⟨52985⟩⟩
def transferEvent : Nat := 111701
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨52983⟩⟩]⟩ [⟨.result 111420 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111420 .coefficient)
      LeftAuthority111419.bound (LeftAuthority111419.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨52983⟩⟩) (rawTerms := some (Proof.Events435.exact111420RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111419.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111419.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority111419.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111419.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound111701

namespace LeftBound111702
def owner : Owner := ⟨.program ⟨257⟩, ⟨52985⟩⟩
def transferEvent : Nat := 111702
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 111697 .summary) (.transfer 111701) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111697 .summary)
      LeftBound111696.bound (LeftBound111696.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨52532⟩⟩) (rawTerms := some (Proof.Events436.exact111697RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 111701)
      LeftBound111701.bound (LeftBound111701.actual selector witness) := by
  exact .transfer (LeftBound111701.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound111696.bound LeftBound111701.bound
def bound : CoeffClass := .finite ⟨32189593014266254325632330629120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111696.bound, LeftBound111701.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound111696.actual selector witness) * (LeftBound111701.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111702

namespace LeftBound111713
def owner : Owner := ⟨.program ⟨257⟩, ⟨51778⟩⟩
def transferEvent : Nat := 111713
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 111711 .coefficient) (.value (.predecessor 1 111712 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111711 .coefficient)
      LeftAuthority111709.bound (LeftAuthority111709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111710RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111712 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority111709.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111709.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111709.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound111713

namespace LeftBound111717
def owner : Owner := ⟨.program ⟨257⟩, ⟨51779⟩⟩
def transferEvent : Nat := 111717
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 111715 .coefficient) (.predecessor 1 111716 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111715 .coefficient)
      LeftBound105242.bound (LeftBound105242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111716 .coefficient)
      LeftBound111713.bound (LeftBound111713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111713.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105242.bound LeftBound111713.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105242.bound, LeftBound111713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105242.actual selector witness) * (LeftBound111713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111717

namespace LeftBound111718
def owner : Owner := ⟨.program ⟨257⟩, ⟨51779⟩⟩
def transferEvent : Nat := 111718
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨51776⟩⟩]⟩ [⟨.result 111710 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111710 .coefficient)
      LeftAuthority111709.bound (LeftAuthority111709.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51776⟩⟩) (rawTerms := some (Proof.Events436.exact111710RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111709.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111709.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority111709.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111709.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound111718

namespace LeftBound111719
def owner : Owner := ⟨.program ⟨257⟩, ⟨51779⟩⟩
def transferEvent : Nat := 111719
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 111718) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 111718)
      LeftBound111718.bound (LeftBound111718.actual selector witness) := by
  exact .transfer (LeftBound111718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound111718.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound111718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound111718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111719

namespace LeftBound111814
def owner : Owner := ⟨.program ⟨257⟩, ⟨50897⟩⟩
def transferEvent : Nat := 111814
def frameStart : Nat := 111775
def rule : BoundRule := .identity (.predecessor 0 111813 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111813 .coefficient)
      LeftAuthority111811.bound (LeftAuthority111811.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111812RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111811.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111811.derived selector witness)

def rawBound : CoeffClass := LeftAuthority111811.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111811.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority111811.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound111814

namespace LeftBound111831
def owner : Owner := ⟨.program ⟨257⟩, ⟨52370⟩⟩
def transferEvent : Nat := 111831
def frameStart : Nat := 111775
def rule : BoundRule := .sum [.predecessor 0 111829 .coefficient, .predecessor 1 111830 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111829 .coefficient)
      LeftBound111814.bound (LeftBound111814.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound111814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111830 .coefficient)
      LeftAuthority111827.bound (LeftAuthority111827.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority111827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111814.bound, LeftAuthority111827.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111814.bound, LeftAuthority111827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111814.actual selector witness, LeftAuthority111827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111831

namespace LeftBound111834
def owner : Owner := ⟨.program ⟨257⟩, ⟨52371⟩⟩
def transferEvent : Nat := 111834
def frameStart : Nat := 111775
def rule : BoundRule := .identity (.predecessor 0 111833 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111833 .coefficient)
      LeftBound111831.bound (LeftBound111831.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound111831.derived selector witness)

def rawBound : CoeffClass := LeftBound111831.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111831.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound111831.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound111834

namespace LeftBound111840
def owner : Owner := ⟨.program ⟨257⟩, ⟨52372⟩⟩
def transferEvent : Nat := 111840
def frameStart : Nat := 111775
def rule : BoundRule := .product (.predecessor 0 111838 .coefficient) (.predecessor 1 111839 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111838 .coefficient)
      LeftAuthority111836.bound (LeftAuthority111836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111839 .coefficient)
      LeftBound111834.bound (LeftBound111834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111834.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority111836.bound LeftBound111834.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111836.bound, LeftBound111834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority111836.actual selector witness) * (LeftBound111834.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111840

namespace LeftBound111848
def owner : Owner := ⟨.program ⟨257⟩, ⟨52373⟩⟩
def transferEvent : Nat := 111848
def frameStart : Nat := 111775
def rule : BoundRule := .sum [.predecessor 0 111846 .coefficient, .predecessor 1 111847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111846 .coefficient)
      LeftAuthority111844.bound (LeftAuthority111844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111847 .coefficient)
      LeftBound111840.bound (LeftBound111840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111840.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111840.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority111844.bound, LeftBound111840.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111844.bound, LeftBound111840.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority111844.actual selector witness, LeftBound111840.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111848

namespace LeftBound111852
def owner : Owner := ⟨.program ⟨257⟩, ⟨52984⟩⟩
def transferEvent : Nat := 111852
def frameStart : Nat := 111775
def rule : BoundRule := .product (.predecessor 0 111850 .coefficient) (.predecessor 1 111851 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111850 .coefficient)
      LeftBound111848.bound (LeftBound111848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111848.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111848.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111851 .coefficient)
      LeftAuthority111825.bound (LeftAuthority111825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111825.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound111848.bound LeftAuthority111825.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111848.bound, LeftAuthority111825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound111848.actual selector witness) * (LeftAuthority111825.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111852

namespace LeftBound111863
def owner : Owner := ⟨.program ⟨257⟩, ⟨51182⟩⟩
def transferEvent : Nat := 111863
def frameStart : Nat := 111775
def rule : BoundRule := .product (.predecessor 0 111861 .coefficient) (.predecessor 1 111862 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111861 .coefficient)
      LeftAuthority111836.bound (LeftAuthority111836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111836.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111836.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111862 .coefficient)
      LeftAuthority111859.bound (LeftAuthority111859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111859.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111859.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority111836.bound LeftAuthority111859.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111836.bound, LeftAuthority111859.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority111836.actual selector witness) * (LeftAuthority111859.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound111863

namespace LeftBound111871
def owner : Owner := ⟨.program ⟨257⟩, ⟨51183⟩⟩
def transferEvent : Nat := 111871
def frameStart : Nat := 111775
def rule : BoundRule := .sum [.predecessor 0 111869 .coefficient, .predecessor 1 111870 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111869 .coefficient)
      LeftAuthority111867.bound (LeftAuthority111867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111870 .coefficient)
      LeftBound111863.bound (LeftBound111863.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111863.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority111867.bound, LeftBound111863.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111867.bound, LeftBound111863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority111867.actual selector witness, LeftBound111863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111871

namespace LeftBound111875
def owner : Owner := ⟨.program ⟨257⟩, ⟨52988⟩⟩
def transferEvent : Nat := 111875
def frameStart : Nat := 111775
def rule : BoundRule := .sum [.predecessor 0 111873 .coefficient, .predecessor 1 111874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111873 .coefficient)
      LeftBound111871.bound (LeftBound111871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact111872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111871.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111874 .coefficient)
      LeftBound111852.bound (LeftBound111852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111857RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111852.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111871.bound, LeftBound111852.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111871.bound, LeftBound111852.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111871.actual selector witness, LeftBound111852.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111875

namespace LeftBound111888
def owner : Owner := ⟨.program ⟨257⟩, ⟨52986⟩⟩
def transferEvent : Nat := 111888
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 111886 .coefficient, .predecessor 1 111887 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 111886 .coefficient)
      LeftBound111717.bound (LeftBound111717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact111885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 111887 .coefficient)
      LeftBound111700.bound (LeftBound111700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events436.exact111707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111700.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111717.bound, LeftBound111700.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111717.bound, LeftBound111700.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111717.actual selector witness, LeftBound111700.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound111888

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
