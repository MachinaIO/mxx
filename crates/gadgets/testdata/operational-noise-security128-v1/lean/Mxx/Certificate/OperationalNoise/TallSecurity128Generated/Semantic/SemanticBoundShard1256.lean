import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1189
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1255

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound187035
def owner : Owner := ⟨.program ⟨257⟩, ⟨71331⟩⟩
def transferEvent : Nat := 187035
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 187030 .summary) (.transfer 187034) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187030 .summary)
      LeftBound187029.bound (LeftBound187029.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70426⟩⟩) (rawTerms := some (Proof.Events730.exact187030RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound187029.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 187034)
      LeftBound187034.bound (LeftBound187034.actual selector witness) := by
  exact .transfer (LeftBound187034.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound187029.bound LeftBound187034.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound187029.bound, LeftBound187034.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound187029.actual selector witness) * (LeftBound187034.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound187035

namespace LeftBound187114
def owner : Owner := ⟨.program ⟨257⟩, ⟨68402⟩⟩
def transferEvent : Nat := 187114
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 187112 .coefficient) (.value (.predecessor 1 187113 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187112 .coefficient)
      LeftAuthority187110.bound (LeftAuthority187110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187110.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187113 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority187110.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority187110.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority187110.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound187114

namespace LeftBound187118
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def transferEvent : Nat := 187118
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 187116 .coefficient) (.predecessor 1 187117 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 187116 .coefficient)
      LeftBound178367.bound (LeftBound178367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound178367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound178367.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 187117 .coefficient)
      LeftBound187114.bound (LeftBound187114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events730.exact187115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound187114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound187114.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178367.bound LeftBound187114.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178367.bound, LeftBound187114.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178367.actual selector witness) * (LeftBound187114.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound187118

namespace LeftBound187119
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def transferEvent : Nat := 187119
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68400⟩⟩]⟩ [⟨.result 187111 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 187111 .coefficient)
      LeftAuthority187110.bound (LeftAuthority187110.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68400⟩⟩) (rawTerms := some (Proof.Events730.exact187111RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187110.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority187110.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority187110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority187110.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound187119

namespace LeftBound187120
def owner : Owner := ⟨.program ⟨257⟩, ⟨68403⟩⟩
def transferEvent : Nat := 187120
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 178370 .summary) (.transfer 187119) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 178370 .summary)
      LeftBound178368.bound (LeftBound178368.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨6186⟩⟩) (rawTerms := some (Proof.Events696.exact178370RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound178368.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 187119)
      LeftBound187119.bound (LeftBound187119.actual selector witness) := by
  exact .transfer (LeftBound187119.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound178368.bound LeftBound187119.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound178368.bound, LeftBound187119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound178368.actual selector witness) * (LeftBound187119.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound187120

namespace LeftBound188148
def owner : Owner := ⟨.program ⟨257⟩, ⟨18924⟩⟩
def transferEvent : Nat := 188148
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188146 .coefficient, .predecessor 1 188147 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188146 .coefficient)
      LeftAuthority188144.bound (LeftAuthority188144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188147 .coefficient)
      LeftAuthority188121.bound (LeftAuthority188121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188121.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority188144.bound, LeftAuthority188121.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority188144.bound, LeftAuthority188121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority188144.actual selector witness, LeftAuthority188121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188148

namespace LeftBound188152
def owner : Owner := ⟨.program ⟨257⟩, ⟨22144⟩⟩
def transferEvent : Nat := 188152
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188150 .coefficient, .predecessor 1 188151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188150 .coefficient)
      LeftBound188148.bound (LeftBound188148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188148.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188151 .coefficient)
      LeftAuthority188098.bound (LeftAuthority188098.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188098.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188098.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188148.bound, LeftAuthority188098.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188148.bound, LeftAuthority188098.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188148.actual selector witness, LeftAuthority188098.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188152

namespace LeftBound188156
def owner : Owner := ⟨.program ⟨257⟩, ⟨32164⟩⟩
def transferEvent : Nat := 188156
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188154 .coefficient, .predecessor 1 188155 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188154 .coefficient)
      LeftBound188152.bound (LeftBound188152.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188153RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188152.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188155 .coefficient)
      LeftAuthority188075.bound (LeftAuthority188075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188152.bound, LeftAuthority188075.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188152.bound, LeftAuthority188075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188152.actual selector witness, LeftAuthority188075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188156

namespace LeftBound188160
def owner : Owner := ⟨.program ⟨257⟩, ⟨51219⟩⟩
def transferEvent : Nat := 188160
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188158 .coefficient, .predecessor 1 188159 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188158 .coefficient)
      LeftBound188156.bound (LeftBound188156.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188156.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188156.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188159 .coefficient)
      LeftAuthority188052.bound (LeftAuthority188052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188052.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188052.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188156.bound, LeftAuthority188052.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188156.bound, LeftAuthority188052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188156.actual selector witness, LeftAuthority188052.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188160

namespace LeftBound188164
def owner : Owner := ⟨.program ⟨257⟩, ⟨54199⟩⟩
def transferEvent : Nat := 188164
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188162 .coefficient, .predecessor 1 188163 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188162 .coefficient)
      LeftBound188160.bound (LeftBound188160.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188160.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188163 .coefficient)
      LeftAuthority188029.bound (LeftAuthority188029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188029.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188029.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188160.bound, LeftAuthority188029.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188160.bound, LeftAuthority188029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188160.actual selector witness, LeftAuthority188029.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188164

namespace LeftBound188168
def owner : Owner := ⟨.program ⟨257⟩, ⟨57179⟩⟩
def transferEvent : Nat := 188168
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188166 .coefficient, .predecessor 1 188167 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188166 .coefficient)
      LeftBound188164.bound (LeftBound188164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188167 .coefficient)
      LeftAuthority188006.bound (LeftAuthority188006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact188007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority188006.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority188006.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188164.bound, LeftAuthority188006.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188164.bound, LeftAuthority188006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188164.actual selector witness, LeftAuthority188006.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188168

namespace LeftBound188172
def owner : Owner := ⟨.program ⟨257⟩, ⟨60159⟩⟩
def transferEvent : Nat := 188172
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188170 .coefficient, .predecessor 1 188171 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188170 .coefficient)
      LeftBound188168.bound (LeftBound188168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188168.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188171 .coefficient)
      LeftAuthority187983.bound (LeftAuthority187983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact187984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187983.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187983.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188168.bound, LeftAuthority187983.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188168.bound, LeftAuthority187983.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188168.actual selector witness, LeftAuthority187983.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188172

namespace LeftBound188176
def owner : Owner := ⟨.program ⟨257⟩, ⟨63139⟩⟩
def transferEvent : Nat := 188176
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188174 .coefficient, .predecessor 1 188175 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188174 .coefficient)
      LeftBound188172.bound (LeftBound188172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188175 .coefficient)
      LeftAuthority187960.bound (LeftAuthority187960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact187961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188172.bound, LeftAuthority187960.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188172.bound, LeftAuthority187960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188172.actual selector witness, LeftAuthority187960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188176

namespace LeftBound188180
def owner : Owner := ⟨.program ⟨257⟩, ⟨66812⟩⟩
def transferEvent : Nat := 188180
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188178 .coefficient, .predecessor 1 188179 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188178 .coefficient)
      LeftBound188176.bound (LeftBound188176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188177RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188176.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188179 .coefficient)
      LeftAuthority187937.bound (LeftAuthority187937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact187938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187937.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187937.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188176.bound, LeftAuthority187937.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188176.bound, LeftAuthority187937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188176.actual selector witness, LeftAuthority187937.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188180

namespace LeftBound188184
def owner : Owner := ⟨.program ⟨257⟩, ⟨66813⟩⟩
def transferEvent : Nat := 188184
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188182 .coefficient, .predecessor 1 188183 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188182 .coefficient)
      LeftBound188180.bound (LeftBound188180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188180.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188183 .coefficient)
      LeftAuthority187914.bound (LeftAuthority187914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events734.exact187915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188180.bound, LeftAuthority187914.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188180.bound, LeftAuthority187914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188180.actual selector witness, LeftAuthority187914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188184

namespace LeftBound188188
def owner : Owner := ⟨.program ⟨257⟩, ⟨66814⟩⟩
def transferEvent : Nat := 188188
def frameStart : Nat := 187711
def rule : BoundRule := .sum [.predecessor 0 188186 .coefficient, .predecessor 1 188187 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 188186 .coefficient)
      LeftBound188184.bound (LeftBound188184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events735.exact188185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound188184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound188184.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 188187 .coefficient)
      LeftAuthority187891.bound (LeftAuthority187891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events733.exact187892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority187891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority187891.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound188184.bound, LeftAuthority187891.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound188184.bound, LeftAuthority187891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound188184.actual selector witness, LeftAuthority187891.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound188188

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
