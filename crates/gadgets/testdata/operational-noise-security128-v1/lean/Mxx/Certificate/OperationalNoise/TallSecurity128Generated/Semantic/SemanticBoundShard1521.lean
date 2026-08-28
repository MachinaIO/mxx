import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1489
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1492
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1494
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1520

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound225826
def owner : Owner := ⟨.program ⟨257⟩, ⟨27139⟩⟩
def transferEvent : Nat := 225826
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨27136⟩⟩]⟩ [⟨.result 225818 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225818 .coefficient)
      LeftAuthority225817.bound (LeftAuthority225817.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨27136⟩⟩) (rawTerms := some (Proof.Events882.exact225818RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225817.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority225817.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority225817.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound225826

namespace LeftBound225827
def owner : Owner := ⟨.program ⟨257⟩, ⟨27139⟩⟩
def transferEvent : Nat := 225827
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 222245 .summary) (.transfer 225826) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 222245 .summary)
      LeftBound222243.bound (LeftBound222243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5581⟩⟩) (rawTerms := some (Proof.Events868.exact222245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound222243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 225826)
      LeftBound225826.bound (LeftBound225826.actual selector witness) := by
  exact .transfer (LeftBound225826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound222243.bound LeftBound225826.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222243.bound, LeftBound225826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound222243.actual selector witness) * (LeftBound225826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225827

namespace LeftBound225922
def owner : Owner := ⟨.program ⟨257⟩, ⟨26401⟩⟩
def transferEvent : Nat := 225922
def frameStart : Nat := 225883
def rule : BoundRule := .identity (.predecessor 0 225921 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225921 .coefficient)
      LeftAuthority225919.bound (LeftAuthority225919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225919.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225919.derived selector witness)

def rawBound : CoeffClass := LeftAuthority225919.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority225919.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound225922

namespace LeftBound225939
def owner : Owner := ⟨.program ⟨257⟩, ⟨27762⟩⟩
def transferEvent : Nat := 225939
def frameStart : Nat := 225883
def rule : BoundRule := .sum [.predecessor 0 225937 .coefficient, .predecessor 1 225938 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225937 .coefficient)
      LeftBound225922.bound (LeftBound225922.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound225922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225938 .coefficient)
      LeftAuthority225935.bound (LeftAuthority225935.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority225935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225922.bound, LeftAuthority225935.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225922.bound, LeftAuthority225935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225922.actual selector witness, LeftAuthority225935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225939

namespace LeftBound225942
def owner : Owner := ⟨.program ⟨257⟩, ⟨27763⟩⟩
def transferEvent : Nat := 225942
def frameStart : Nat := 225883
def rule : BoundRule := .identity (.predecessor 0 225941 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225941 .coefficient)
      LeftBound225939.bound (LeftBound225939.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound225939.derived selector witness)

def rawBound : CoeffClass := LeftBound225939.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound225939.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound225942

namespace LeftBound225948
def owner : Owner := ⟨.program ⟨257⟩, ⟨27764⟩⟩
def transferEvent : Nat := 225948
def frameStart : Nat := 225883
def rule : BoundRule := .product (.predecessor 0 225946 .coefficient) (.predecessor 1 225947 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225946 .coefficient)
      LeftAuthority225944.bound (LeftAuthority225944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225947 .coefficient)
      LeftBound225942.bound (LeftBound225942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225942.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority225944.bound LeftBound225942.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225944.bound, LeftBound225942.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority225944.actual selector witness) * (LeftBound225942.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225948

namespace LeftBound225956
def owner : Owner := ⟨.program ⟨257⟩, ⟨27765⟩⟩
def transferEvent : Nat := 225956
def frameStart : Nat := 225883
def rule : BoundRule := .sum [.predecessor 0 225954 .coefficient, .predecessor 1 225955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225954 .coefficient)
      LeftAuthority225952.bound (LeftAuthority225952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225952.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225955 .coefficient)
      LeftBound225948.bound (LeftBound225948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225948.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority225952.bound, LeftBound225948.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225952.bound, LeftBound225948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority225952.actual selector witness, LeftBound225948.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225956

namespace LeftBound225960
def owner : Owner := ⟨.program ⟨257⟩, ⟨28265⟩⟩
def transferEvent : Nat := 225960
def frameStart : Nat := 225883
def rule : BoundRule := .product (.predecessor 0 225958 .coefficient) (.predecessor 1 225959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225958 .coefficient)
      LeftBound225956.bound (LeftBound225956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225959 .coefficient)
      LeftAuthority225933.bound (LeftAuthority225933.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225934RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225933.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225933.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound225956.bound LeftAuthority225933.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225956.bound, LeftAuthority225933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound225956.actual selector witness) * (LeftAuthority225933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225960

namespace LeftBound225971
def owner : Owner := ⟨.program ⟨257⟩, ⟨26607⟩⟩
def transferEvent : Nat := 225971
def frameStart : Nat := 225883
def rule : BoundRule := .product (.predecessor 0 225969 .coefficient) (.predecessor 1 225970 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225969 .coefficient)
      LeftAuthority225944.bound (LeftAuthority225944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225944.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225970 .coefficient)
      LeftAuthority225967.bound (LeftAuthority225967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority225944.bound LeftAuthority225967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225944.bound, LeftAuthority225967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority225944.actual selector witness) * (LeftAuthority225967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound225971

namespace LeftBound225979
def owner : Owner := ⟨.program ⟨257⟩, ⟨26608⟩⟩
def transferEvent : Nat := 225979
def frameStart : Nat := 225883
def rule : BoundRule := .sum [.predecessor 0 225977 .coefficient, .predecessor 1 225978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225977 .coefficient)
      LeftAuthority225975.bound (LeftAuthority225975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority225975.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority225975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225978 .coefficient)
      LeftBound225971.bound (LeftBound225971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority225975.bound, LeftBound225971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority225975.bound, LeftBound225971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority225975.actual selector witness, LeftBound225971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225979

namespace LeftBound225983
def owner : Owner := ⟨.program ⟨257⟩, ⟨28268⟩⟩
def transferEvent : Nat := 225983
def frameStart : Nat := 225883
def rule : BoundRule := .sum [.predecessor 0 225981 .coefficient, .predecessor 1 225982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225981 .coefficient)
      LeftBound225979.bound (LeftBound225979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225982 .coefficient)
      LeftBound225960.bound (LeftBound225960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225979.bound, LeftBound225960.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225979.bound, LeftBound225960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225979.actual selector witness, LeftBound225960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225983

namespace LeftBound225996
def owner : Owner := ⟨.program ⟨257⟩, ⟨28267⟩⟩
def transferEvent : Nat := 225996
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 225994 .coefficient, .predecessor 1 225995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 225994 .coefficient)
      LeftBound225825.bound (LeftBound225825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225825.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 225995 .coefficient)
      LeftBound225808.bound (LeftBound225808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact225815RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound225808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound225808.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225825.bound, LeftBound225808.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225825.bound, LeftBound225808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225825.actual selector witness, LeftBound225808.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225996

namespace LeftBound225999
def owner : Owner := ⟨.program ⟨257⟩, ⟨28267⟩⟩
def transferEvent : Nat := 225999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 225993 .summary, .result 225815 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225993 .summary)
      LeftBound225827.bound (LeftBound225827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨27139⟩⟩) (rawTerms := some (Proof.Events882.exact225993RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound225827.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 225815 .summary)
      LeftBound225810.bound (LeftBound225810.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨28266⟩⟩) (rawTerms := some (Proof.Events882.exact225815RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound225810.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound225827.bound, LeftBound225810.bound]
def bound : CoeffClass := .finite ⟨32191557518723330170883082027008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound225827.bound, LeftBound225810.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound225827.actual selector witness, LeftBound225810.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound225999

namespace LeftBound226023
def owner : Owner := ⟨.program ⟨257⟩, ⟨25719⟩⟩
def transferEvent : Nat := 226023
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 226021 .coefficient) (.predecessor 1 226022 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226021 .coefficient)
      LeftAuthority10749.bound (LeftAuthority10749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10750RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10749.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226022 .coefficient)
      LeftBound222151.bound (LeftBound222151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222151.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority10749.bound LeftBound222151.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10749.bound, LeftBound222151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority10749.actual selector witness) * (LeftBound222151.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound226023

namespace LeftBound226028
def owner : Owner := ⟨.program ⟨257⟩, ⟨8468⟩⟩
def transferEvent : Nat := 226028
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 226026 .coefficient) (.predecessor 1 226027 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226026 .coefficient)
      LeftBound222022.bound (LeftBound222022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events867.exact222023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound222022.bound, RecordedBoundRefines] <;> decide)
      (LeftBound222022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226027 .coefficient)
      LeftBound21087.bound (LeftBound21087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound222022.bound LeftBound21087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound222022.bound, LeftBound21087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound222022.actual selector witness) * (LeftBound21087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound226028

namespace LeftBound226033
def owner : Owner := ⟨.program ⟨257⟩, ⟨25720⟩⟩
def transferEvent : Nat := 226033
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 226031 .coefficient, .predecessor 1 226032 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 226031 .coefficient)
      LeftBound226028.bound (LeftBound226028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact226030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 226032 .coefficient)
      LeftBound226023.bound (LeftBound226023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events882.exact226025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound226023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound226023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound226028.bound, LeftBound226023.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound226028.bound, LeftBound226023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound226028.actual selector witness, LeftBound226023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound226033

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
