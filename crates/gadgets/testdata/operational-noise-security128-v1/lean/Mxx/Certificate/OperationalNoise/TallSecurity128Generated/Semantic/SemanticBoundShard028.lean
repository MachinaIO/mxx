import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard001
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard027

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8938
def owner : Owner := ⟨.program ⟨257⟩, ⟨22139⟩⟩
def transferEvent : Nat := 8938
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8936 .coefficient) (.predecessor 1 8937 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8936 .coefficient)
      LeftAuthority8934.bound (LeftAuthority8934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8937 .coefficient)
      LeftAuthority692.bound (LeftAuthority692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority692.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8934.bound LeftAuthority692.bound
def bound : CoeffClass := .finite ⟨187661410175051153573232, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8934.bound, LeftAuthority692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority8934.actual selector witness) * (LeftAuthority692.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8938

namespace LeftBound8946
def owner : Owner := ⟨.program ⟨257⟩, ⟨18919⟩⟩
def transferEvent : Nat := 8946
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8944 .coefficient) (.predecessor 1 8945 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8944 .coefficient)
      LeftAuthority8942.bound (LeftAuthority8942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8945 .coefficient)
      LeftAuthority702.bound (LeftAuthority702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact703RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority702.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority702.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8942.bound LeftAuthority702.bound
def bound : CoeffClass := .finite ⟨175932572039110456474905, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8942.bound, LeftAuthority702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority8942.actual selector witness) * (LeftAuthority702.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8946

namespace LeftBound8954
def owner : Owner := ⟨.program ⟨257⟩, ⟨16079⟩⟩
def transferEvent : Nat := 8954
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8952 .coefficient) (.predecessor 1 8953 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8952 .coefficient)
      LeftAuthority8950.bound (LeftAuthority8950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8953 .coefficient)
      LeftAuthority712.bound (LeftAuthority712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8950.bound LeftAuthority712.bound
def bound : CoeffClass := .finite ⟨156384508479209294644360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8950.bound, LeftAuthority712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority8950.actual selector witness) * (LeftAuthority712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8954

namespace LeftBound8959
def owner : Owner := ⟨.program ⟨257⟩, ⟨16080⟩⟩
def transferEvent : Nat := 8959
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8957 .coefficient, .predecessor 1 8958 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8957 .coefficient)
      LeftBound726.bound (LeftBound726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound726.bound, RecordedBoundRefines] <;> decide)
      (LeftBound726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8958 .coefficient)
      LeftBound8954.bound (LeftBound8954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound726.bound, LeftBound8954.bound]
def bound : CoeffClass := .finite ⟨156384508479209294644362, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound726.bound, LeftBound8954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound726.actual selector witness, LeftBound8954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8959

namespace LeftBound8963
def owner : Owner := ⟨.program ⟨257⟩, ⟨18920⟩⟩
def transferEvent : Nat := 8963
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8961 .coefficient, .predecessor 1 8962 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8961 .coefficient)
      LeftBound8959.bound (LeftBound8959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8959.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8962 .coefficient)
      LeftBound8946.bound (LeftBound8946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8959.bound, LeftBound8946.bound]
def bound : CoeffClass := .finite ⟨332317080518319751119267, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8959.bound, LeftBound8946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8959.actual selector witness, LeftBound8946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8963

namespace LeftBound8967
def owner : Owner := ⟨.program ⟨257⟩, ⟨22140⟩⟩
def transferEvent : Nat := 8967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8965 .coefficient, .predecessor 1 8966 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8965 .coefficient)
      LeftBound8963.bound (LeftBound8963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8963.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8966 .coefficient)
      LeftBound8938.bound (LeftBound8938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8938.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8963.bound, LeftBound8938.bound]
def bound : CoeffClass := .finite ⟨519978490693370904692499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8963.bound, LeftBound8938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8963.actual selector witness, LeftBound8938.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8967

namespace LeftBound8971
def owner : Owner := ⟨.program ⟨257⟩, ⟨32160⟩⟩
def transferEvent : Nat := 8971
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8969 .coefficient, .predecessor 1 8970 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8969 .coefficient)
      LeftBound8967.bound (LeftBound8967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8967.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8970 .coefficient)
      LeftBound8930.bound (LeftBound8930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8967.bound, LeftBound8930.bound]
def bound : CoeffClass := .finite ⟨721044287309497140663819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8967.bound, LeftBound8930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8967.actual selector witness, LeftBound8930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8971

namespace LeftBound8975
def owner : Owner := ⟨.program ⟨257⟩, ⟨51224⟩⟩
def transferEvent : Nat := 8975
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8973 .coefficient, .predecessor 1 8974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8973 .coefficient)
      LeftBound8971.bound (LeftBound8971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8974 .coefficient)
      LeftBound8922.bound (LeftBound8922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8971.bound, LeftBound8922.bound]
def bound : CoeffClass := .finite ⟨934295889781146178815219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8971.bound, LeftBound8922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8971.actual selector witness, LeftBound8922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8975

namespace LeftBound8979
def owner : Owner := ⟨.program ⟨257⟩, ⟨54204⟩⟩
def transferEvent : Nat := 8979
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8977 .coefficient, .predecessor 1 8978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8977 .coefficient)
      LeftBound8975.bound (LeftBound8975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8978 .coefficient)
      LeftBound8914.bound (LeftBound8914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8975.bound, LeftBound8914.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8975.bound, LeftBound8914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8975.actual selector witness, LeftBound8914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8979

namespace LeftBound8983
def owner : Owner := ⟨.program ⟨257⟩, ⟨57184⟩⟩
def transferEvent : Nat := 8983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8981 .coefficient, .predecessor 1 8982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8981 .coefficient)
      LeftBound8979.bound (LeftBound8979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8982 .coefficient)
      LeftBound8906.bound (LeftBound8906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8979.bound, LeftBound8906.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8979.bound, LeftBound8906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8979.actual selector witness, LeftBound8906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8983

namespace LeftBound8987
def owner : Owner := ⟨.program ⟨257⟩, ⟨60164⟩⟩
def transferEvent : Nat := 8987
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8985 .coefficient, .predecessor 1 8986 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8985 .coefficient)
      LeftBound8983.bound (LeftBound8983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8986 .coefficient)
      LeftBound8898.bound (LeftBound8898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8983.bound, LeftBound8898.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8983.bound, LeftBound8898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8983.actual selector witness, LeftBound8898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8987

namespace LeftBound8991
def owner : Owner := ⟨.program ⟨257⟩, ⟨63144⟩⟩
def transferEvent : Nat := 8991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8989 .coefficient, .predecessor 1 8990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8989 .coefficient)
      LeftBound8987.bound (LeftBound8987.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8987.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8987.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8990 .coefficient)
      LeftBound8890.bound (LeftBound8890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8987.bound, LeftBound8890.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8987.bound, LeftBound8890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8987.actual selector witness, LeftBound8890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8991

namespace LeftBound8995
def owner : Owner := ⟨.program ⟨257⟩, ⟨66800⟩⟩
def transferEvent : Nat := 8995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8993 .coefficient, .predecessor 1 8994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8993 .coefficient)
      LeftBound8991.bound (LeftBound8991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8994 .coefficient)
      LeftBound8882.bound (LeftBound8882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8991.bound, LeftBound8882.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8991.bound, LeftBound8882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8991.actual selector witness, LeftBound8882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8995

namespace LeftBound8999
def owner : Owner := ⟨.program ⟨257⟩, ⟨66801⟩⟩
def transferEvent : Nat := 8999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 8997 .coefficient, .predecessor 1 8998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 8997 .coefficient)
      LeftBound8995.bound (LeftBound8995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 8998 .coefficient)
      LeftBound8874.bound (LeftBound8874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8995.bound, LeftBound8874.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8995.bound, LeftBound8874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8995.actual selector witness, LeftBound8874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8999

namespace LeftBound9003
def owner : Owner := ⟨.program ⟨257⟩, ⟨66802⟩⟩
def transferEvent : Nat := 9003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9001 .coefficient, .predecessor 1 9002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9001 .coefficient)
      LeftBound8999.bound (LeftBound8999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9002 .coefficient)
      LeftBound8866.bound (LeftBound8866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8999.bound, LeftBound8866.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8999.bound, LeftBound8866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound8999.actual selector witness, LeftBound8866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9003

namespace LeftBound9007
def owner : Owner := ⟨.program ⟨257⟩, ⟨66803⟩⟩
def transferEvent : Nat := 9007
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9005 .coefficient, .predecessor 1 9006 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 9005 .coefficient)
      LeftBound9003.bound (LeftBound9003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 9006 .coefficient)
      LeftBound8858.bound (LeftBound8858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9003.bound, LeftBound8858.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9003.bound, LeftBound8858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound9003.actual selector witness, LeftBound8858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9007

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
