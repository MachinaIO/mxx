import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard048

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8996
def owner : Owner := ⟨.program ⟨214⟩, ⟨12408⟩⟩
def transferEvent : Nat := 8996
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9840⟩⟩], []⟩ [⟨.result 169 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 169 .coefficient)
      LeftAuthority168.bound (LeftAuthority168.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9840⟩⟩) (rawTerms := some (Proof.Events000.exact169RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority168.bound []
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority168.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8996

namespace LeftBound8997
def owner : Owner := ⟨.program ⟨214⟩, ⟨12408⟩⟩
def transferEvent : Nat := 8997
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8992 .summary) (.transfer 8996) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8992 .summary)
      LeftBound8990.bound (LeftBound8990.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12407⟩⟩) (rawTerms := some (Proof.Events035.exact8992RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8996)
      LeftBound8996.bound (LeftBound8996.actual selector witness) := by
  exact .transfer (LeftBound8996.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound8990.bound LeftBound8996.bound
def bound : CoeffClass := .finite ⟨33280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8990.bound, LeftBound8996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound8990.actual selector witness) * (LeftBound8996.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8997

namespace LeftBound9006
def owner : Owner := ⟨.program ⟨214⟩, ⟨7868⟩⟩
def transferEvent : Nat := 9006
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 9004 .coefficient) (.value (.predecessor 1 9005 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9004 .coefficient)
      LeftAuthority9002.bound (LeftAuthority9002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9005 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority9002.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9002.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9002.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9006

namespace LeftBound9009
def owner : Owner := ⟨.program ⟨214⟩, ⟨79⟩⟩
def transferEvent : Nat := 9009
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9008 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9008 .coefficient)
      LeftAuthority6440.bound (LeftAuthority6440.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6441RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6440.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6440.derived selector witness)

def rawBound : CoeffClass := LeftAuthority6440.bound
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority6440.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9009

namespace LeftBound9013
def owner : Owner := ⟨.program ⟨214⟩, ⟨9841⟩⟩
def transferEvent : Nat := 9013
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 9011 .coefficient) (.predecessor 1 9012 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9011 .coefficient)
      LeftAuthority168.bound (LeftAuthority168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9012 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority168.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority168.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound9013

namespace LeftBound9017
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 9017
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 9016 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9016 .coefficient)
      LeftAuthority5869.bound (LeftAuthority5869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5869.derived selector witness)

def rawBound : CoeffClass := LeftAuthority5869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority5869.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound9017

namespace LeftBound9021
def owner : Owner := ⟨.program ⟨214⟩, ⟨7373⟩⟩
def transferEvent : Nat := 9021
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9019 .coefficient) (.predecessor 1 9020 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9019 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9020 .coefficient)
      LeftBound9017.bound (LeftBound9017.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9017.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound9017.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound9017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound9017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9021

namespace LeftBound9026
def owner : Owner := ⟨.program ⟨214⟩, ⟨9842⟩⟩
def transferEvent : Nat := 9026
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9024 .coefficient, .predecessor 1 9025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9024 .coefficient)
      LeftBound9021.bound (LeftBound9021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9021.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9025 .coefficient)
      LeftBound9013.bound (LeftBound9013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9015RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9013.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9021.bound, LeftBound9013.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9021.bound, LeftBound9013.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9021.actual selector witness, LeftBound9013.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9026

namespace LeftBound9030
def owner : Owner := ⟨.program ⟨214⟩, ⟨9843⟩⟩
def transferEvent : Nat := 9030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9028 .coefficient, .predecessor 1 9029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9028 .coefficient)
      LeftBound9026.bound (LeftBound9026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9029 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9026.bound, LeftBound9009.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9026.bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9026.actual selector witness, LeftBound9009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9030

namespace LeftBound9031
def owner : Owner := ⟨.program ⟨214⟩, ⟨9843⟩⟩
def transferEvent : Nat := 9031
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨79⟩⟩]⟩ [⟨.result 9010 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9010 .coefficient)
      LeftBound9009.bound (LeftBound9009.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨79⟩⟩) (rawTerms := some (Proof.Events035.exact9010RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9009.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9009.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9009.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9031

namespace LeftBound9036
def owner : Owner := ⟨.program ⟨214⟩, ⟨9844⟩⟩
def transferEvent : Nat := 9036
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9034 .coefficient) (.predecessor 1 9035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9034 .coefficient)
      LeftBound9030.bound (LeftBound9030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9035 .coefficient)
      LeftBound9006.bound (LeftBound9006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9006.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9030.bound LeftBound9006.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9030.bound, LeftBound9006.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9030.actual selector witness) * (LeftBound9006.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9036

namespace LeftBound9037
def owner : Owner := ⟨.program ⟨214⟩, ⟨9844⟩⟩
def transferEvent : Nat := 9037
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7867⟩⟩]⟩ [⟨.result 9003 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9003 .coefficient)
      LeftAuthority9002.bound (LeftAuthority9002.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7867⟩⟩) (rawTerms := some (Proof.Events035.exact9003RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9002.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9002.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9002.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9002.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9002.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound9037

namespace LeftBound9038
def owner : Owner := ⟨.program ⟨214⟩, ⟨9844⟩⟩
def transferEvent : Nat := 9038
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9033 .summary) (.transfer 9037) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9033 .summary)
      LeftBound9031.bound (LeftBound9031.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9843⟩⟩) (rawTerms := some (Proof.Events035.exact9033RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 9037)
      LeftBound9037.bound (LeftBound9037.actual selector witness) := by
  exact .transfer (LeftBound9037.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9031.bound LeftBound9037.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9031.bound, LeftBound9037.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9031.actual selector witness) * (LeftBound9037.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9038

namespace LeftBound9046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12409⟩⟩
def transferEvent : Nat := 9046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 9044 .coefficient, .predecessor 1 9045 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9044 .coefficient)
      LeftBound9036.bound (LeftBound9036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9045 .coefficient)
      LeftBound8995.bound (LeftBound8995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9036.bound, LeftBound8995.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9036.bound, LeftBound8995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9036.actual selector witness, LeftBound8995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9046

namespace LeftBound9048
def owner : Owner := ⟨.program ⟨214⟩, ⟨12409⟩⟩
def transferEvent : Nat := 9048
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 9043 .summary, .result 9000 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9043 .summary)
      LeftBound9038.bound (LeftBound9038.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9844⟩⟩) (rawTerms := some (Proof.Events035.exact9043RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9000 .summary)
      LeftBound8997.bound (LeftBound8997.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12408⟩⟩) (rawTerms := some (Proof.Events035.exact9000RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8997.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound9038.bound, LeftBound8997.bound]
def bound : CoeffClass := .finite ⟨95453696, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9038.bound, LeftBound8997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound9038.actual selector witness, LeftBound8997.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound9048

namespace LeftBound9052
def owner : Owner := ⟨.program ⟨214⟩, ⟨25394⟩⟩
def transferEvent : Nat := 9052
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 9050 .coefficient) (.predecessor 1 9051 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 9050 .coefficient)
      LeftBound9046.bound (LeftBound9046.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact9049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9046.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 9051 .coefficient)
      LeftAuthority8965.bound (LeftAuthority8965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events035.exact8966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9046.bound LeftAuthority8965.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9046.bound, LeftAuthority8965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9046.actual selector witness) * (LeftAuthority8965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound9052

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
