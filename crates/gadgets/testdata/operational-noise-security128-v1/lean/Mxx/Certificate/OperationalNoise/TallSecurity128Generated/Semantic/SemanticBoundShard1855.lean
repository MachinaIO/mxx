import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1794
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1796
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1854

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound273727
def owner : Owner := ⟨.program ⟨257⟩, ⟨23618⟩⟩
def transferEvent : Nat := 273727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273725 .coefficient, .predecessor 1 273726 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273725 .coefficient)
      LeftBound273556.bound (LeftBound273556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273726 .coefficient)
      LeftBound273539.bound (LeftBound273539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1068.exact273546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273556.bound, LeftBound273539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273556.bound, LeftBound273539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273556.actual selector witness, LeftBound273539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273727

namespace LeftBound273730
def owner : Owner := ⟨.program ⟨257⟩, ⟨23618⟩⟩
def transferEvent : Nat := 273730
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 273724 .summary, .result 273546 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273724 .summary)
      LeftBound273558.bound (LeftBound273558.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22513⟩⟩) (rawTerms := some (Proof.Events1069.exact273724RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273546 .summary)
      LeftBound273541.bound (LeftBound273541.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23617⟩⟩) (rawTerms := some (Proof.Events1068.exact273546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273558.bound, LeftBound273541.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273558.bound, LeftBound273541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273558.actual selector witness, LeftBound273541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273730

namespace LeftBound273754
def owner : Owner := ⟨.program ⟨257⟩, ⟨18077⟩⟩
def transferEvent : Nat := 273754
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 273752 .coefficient) (.predecessor 1 273753 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273752 .coefficient)
      LeftAuthority13177.bound (LeftAuthority13177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273753 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13177.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13177.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13177.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound273754

namespace LeftBound273759
def owner : Owner := ⟨.program ⟨257⟩, ⟨7661⟩⟩
def transferEvent : Nat := 273759
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273757 .coefficient) (.predecessor 1 273758 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273757 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273758 .coefficient)
      LeftBound25095.bound (LeftBound25095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound25095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound25095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound25095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273759

namespace LeftBound273764
def owner : Owner := ⟨.program ⟨257⟩, ⟨18078⟩⟩
def transferEvent : Nat := 273764
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273762 .coefficient, .predecessor 1 273763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273762 .coefficient)
      LeftBound273759.bound (LeftBound273759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273763 .coefficient)
      LeftBound273754.bound (LeftBound273754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273759.bound, LeftBound273754.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273759.bound, LeftBound273754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273759.actual selector witness, LeftBound273754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273764

namespace LeftBound273768
def owner : Owner := ⟨.program ⟨257⟩, ⟨18079⟩⟩
def transferEvent : Nat := 273768
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273766 .coefficient, .predecessor 1 273767 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273766 .coefficient)
      LeftBound273764.bound (LeftBound273764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273767 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273764.bound, LeftBound25087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273764.bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273764.actual selector witness, LeftBound25087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273768

namespace LeftBound273769
def owner : Owner := ⟨.program ⟨257⟩, ⟨18079⟩⟩
def transferEvent : Nat := 273769
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩ [⟨.result 25088 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25088 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨131⟩⟩) (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25087.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25087.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound273769

namespace LeftBound273774
def owner : Owner := ⟨.program ⟨257⟩, ⟨18080⟩⟩
def transferEvent : Nat := 273774
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273772 .coefficient) (.predecessor 1 273773 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273772 .coefficient)
      LeftBound273768.bound (LeftBound273768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273773 .coefficient)
      LeftAuthority13180.bound (LeftAuthority13180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13180.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound273768.bound LeftAuthority13180.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273768.bound, LeftAuthority13180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound273768.actual selector witness) * (LeftAuthority13180.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273774

namespace LeftBound273775
def owner : Owner := ⟨.program ⟨257⟩, ⟨18080⟩⟩
def transferEvent : Nat := 273775
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨12556⟩⟩], []⟩ [⟨.result 13181 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 13181 .coefficient)
      LeftAuthority13180.bound (LeftAuthority13180.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨12556⟩⟩) (rawTerms := some (Proof.Events051.exact13181RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13180.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13180.bound []
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority13180.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound273775

namespace LeftBound273776
def owner : Owner := ⟨.program ⟨257⟩, ⟨18080⟩⟩
def transferEvent : Nat := 273776
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 273771 .summary) (.transfer 273775) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 273771 .summary)
      LeftBound273769.bound (LeftBound273769.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨18079⟩⟩) (rawTerms := some (Proof.Events1069.exact273771RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound273769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 273775)
      LeftBound273775.bound (LeftBound273775.actual selector witness) := by
  exact .transfer (LeftBound273775.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound273769.bound LeftBound273775.bound
def bound : CoeffClass := .finite ⟨2555904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273769.bound, LeftBound273775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound273769.actual selector witness) * (LeftBound273775.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273776

namespace LeftBound273782
def owner : Owner := ⟨.program ⟨257⟩, ⟨12557⟩⟩
def transferEvent : Nat := 273782
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 273780 .coefficient) (.predecessor 1 273781 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273780 .coefficient)
      LeftAuthority13180.bound (LeftAuthority13180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273781 .coefficient)
      LeftBound266026.bound (LeftBound266026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1039.exact266028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound266026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound266026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority13180.bound LeftBound266026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13180.bound, LeftBound266026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority13180.actual selector witness) * (LeftBound266026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound273782

namespace LeftBound273787
def owner : Owner := ⟨.program ⟨257⟩, ⟨7633⟩⟩
def transferEvent : Nat := 273787
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273785 .coefficient) (.predecessor 1 273786 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273785 .coefficient)
      LeftBound265897.bound (LeftBound265897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273786 .coefficient)
      LeftBound25136.bound (LeftBound25136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25136.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25136.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound265897.bound LeftBound25136.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound265897.bound, LeftBound25136.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound265897.actual selector witness) * (LeftBound25136.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273787

namespace LeftBound273792
def owner : Owner := ⟨.program ⟨257⟩, ⟨12558⟩⟩
def transferEvent : Nat := 273792
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273790 .coefficient, .predecessor 1 273791 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273790 .coefficient)
      LeftBound273787.bound (LeftBound273787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273791 .coefficient)
      LeftBound273782.bound (LeftBound273782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273787.bound, LeftBound273782.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273787.bound, LeftBound273782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273787.actual selector witness, LeftBound273782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273792

namespace LeftBound273796
def owner : Owner := ⟨.program ⟨257⟩, ⟨12559⟩⟩
def transferEvent : Nat := 273796
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 273794 .coefficient, .predecessor 1 273795 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273794 .coefficient)
      LeftBound273792.bound (LeftBound273792.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273793RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273792.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273792.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273795 .coefficient)
      LeftBound25128.bound (LeftBound25128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound273792.bound, LeftBound25128.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273792.bound, LeftBound25128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound273792.actual selector witness, LeftBound25128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound273796

namespace LeftBound273797
def owner : Owner := ⟨.program ⟨257⟩, ⟨12559⟩⟩
def transferEvent : Nat := 273797
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨103⟩⟩]⟩ [⟨.result 25129 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25129 .coefficient)
      LeftBound25128.bound (LeftBound25128.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨103⟩⟩) (rawTerms := some (Proof.Events098.exact25129RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25128.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25128.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25128.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound273797

namespace LeftBound273802
def owner : Owner := ⟨.program ⟨257⟩, ⟨12560⟩⟩
def transferEvent : Nat := 273802
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 273800 .coefficient) (.predecessor 1 273801 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 273800 .coefficient)
      LeftBound273796.bound (LeftBound273796.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1069.exact273799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound273796.bound, RecordedBoundRefines] <;> decide)
      (LeftBound273796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 273801 .coefficient)
      LeftBound25125.bound (LeftBound25125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound273796.bound LeftBound25125.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound273796.bound, LeftBound25125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound273796.actual selector witness) * (LeftBound25125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound273802

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
