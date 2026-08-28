import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard017
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound5977
def owner : Owner := ⟨.program ⟨214⟩, ⟨7917⟩⟩
def transferEvent : Nat := 5977
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5975 .coefficient) (.predecessor 1 5976 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5975 .coefficient)
      LeftBound5972.bound (LeftBound5972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5972.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5976 .coefficient)
      LeftBound5475.bound (LeftBound5475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound5972.bound LeftBound5475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound5972.bound, LeftBound5475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound5972.actual selector witness) * (LeftBound5475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5977

namespace LeftBound5982
def owner : Owner := ⟨.program ⟨214⟩, ⟨6615⟩⟩
def transferEvent : Nat := 5982
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 5980 .coefficient) (.predecessor 1 5981 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5980 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5981 .coefficient)
      LeftAuthority828.bound (LeftAuthority828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority828.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority828.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority828.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound5982

namespace LeftBound5990
def owner : Owner := ⟨.program ⟨214⟩, ⟨6688⟩⟩
def transferEvent : Nat := 5990
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5988 .coefficient) (.value (.predecessor 1 5989 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5988 .coefficient)
      LeftAuthority5986.bound (LeftAuthority5986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5989 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5986.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5986.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5986.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound5990

namespace LeftBound6000
def owner : Owner := ⟨.program ⟨214⟩, ⟨7822⟩⟩
def transferEvent : Nat := 6000
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 5998 .coefficient) (.value (.predecessor 1 5999 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 5998 .coefficient)
      LeftAuthority5996.bound (LeftAuthority5996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 5999 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority5996.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5996.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5996.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6000

namespace LeftBound6007
def owner : Owner := ⟨.program ⟨214⟩, ⟨7888⟩⟩
def transferEvent : Nat := 6007
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6005 .coefficient) (.predecessor 1 6006 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6005 .coefficient)
      LeftAuthority6003.bound (LeftAuthority6003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6006 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority6003.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6003.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority6003.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6007

namespace LeftBound6012
def owner : Owner := ⟨.program ⟨214⟩, ⟨7912⟩⟩
def transferEvent : Nat := 6012
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6010 .coefficient) (.predecessor 1 6011 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6010 .coefficient)
      LeftBound6007.bound (LeftBound6007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6011 .coefficient)
      LeftBound6000.bound (LeftBound6000.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6001RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6000.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6007.bound LeftBound6000.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6007.bound, LeftBound6000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6007.actual selector witness) * (LeftBound6000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6012

namespace LeftBound6017
def owner : Owner := ⟨.program ⟨214⟩, ⟨7918⟩⟩
def transferEvent : Nat := 6017
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6015 .coefficient) (.predecessor 1 6016 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6015 .coefficient)
      LeftBound6012.bound (LeftBound6012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6016 .coefficient)
      LeftBound5990.bound (LeftBound5990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5990.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6012.bound LeftBound5990.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6012.bound, LeftBound5990.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6012.actual selector witness) * (LeftBound5990.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6017

namespace LeftBound6022
def owner : Owner := ⟨.program ⟨214⟩, ⟨6598⟩⟩
def transferEvent : Nat := 6022
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6020 .coefficient) (.predecessor 1 6021 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6020 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6021 .coefficient)
      LeftAuthority1576.bound (LeftAuthority1576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events006.exact1577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1576.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority1576.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority1576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority1576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6022

namespace LeftBound6030
def owner : Owner := ⟨.program ⟨214⟩, ⟨6654⟩⟩
def transferEvent : Nat := 6030
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6028 .coefficient) (.value (.predecessor 1 6029 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6028 .coefficient)
      LeftAuthority6026.bound (LeftAuthority6026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6029 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6026.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6026.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6026.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6030

namespace LeftBound6040
def owner : Owner := ⟨.program ⟨214⟩, ⟨7824⟩⟩
def transferEvent : Nat := 6040
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6038 .coefficient) (.value (.predecessor 1 6039 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6038 .coefficient)
      LeftAuthority6036.bound (LeftAuthority6036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6036.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6039 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6036.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6036.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6036.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6040

namespace LeftBound6047
def owner : Owner := ⟨.program ⟨214⟩, ⟨7889⟩⟩
def transferEvent : Nat := 6047
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6045 .coefficient) (.predecessor 1 6046 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6045 .coefficient)
      LeftAuthority6043.bound (LeftAuthority6043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6043.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6046 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftAuthority6043.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6043.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftAuthority6043.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6047

namespace LeftBound6052
def owner : Owner := ⟨.program ⟨214⟩, ⟨7913⟩⟩
def transferEvent : Nat := 6052
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6050 .coefficient) (.predecessor 1 6051 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6050 .coefficient)
      LeftBound6047.bound (LeftBound6047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6051 .coefficient)
      LeftBound6040.bound (LeftBound6040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6040.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6047.bound LeftBound6040.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6047.bound, LeftBound6040.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6047.actual selector witness) * (LeftBound6040.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6052

namespace LeftBound6057
def owner : Owner := ⟨.program ⟨214⟩, ⟨7919⟩⟩
def transferEvent : Nat := 6057
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6055 .coefficient) (.predecessor 1 6056 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6055 .coefficient)
      LeftBound6052.bound (LeftBound6052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6052.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6056 .coefficient)
      LeftBound6030.bound (LeftBound6030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6030.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound6052.bound LeftBound6030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6052.bound, LeftBound6030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound6052.actual selector witness) * (LeftBound6030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6057

namespace LeftBound6062
def owner : Owner := ⟨.program ⟨214⟩, ⟨6609⟩⟩
def transferEvent : Nat := 6062
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6060 .coefficient) (.predecessor 1 6061 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6060 .coefficient)
      LeftAuthority1.bound (LeftAuthority1.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact2RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6061 .coefficient)
      LeftAuthority2324.bound (LeftAuthority2324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority1.bound LeftAuthority2324.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1.bound, LeftAuthority2324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority1.actual selector witness) * (LeftAuthority2324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6062

namespace LeftBound6070
def owner : Owner := ⟨.program ⟨214⟩, ⟨6676⟩⟩
def transferEvent : Nat := 6070
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6068 .coefficient) (.value (.predecessor 1 6069 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6068 .coefficient)
      LeftAuthority6066.bound (LeftAuthority6066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6066.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6066.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6069 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6066.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6066.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6066.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6070

namespace LeftBound6080
def owner : Owner := ⟨.program ⟨214⟩, ⟨7826⟩⟩
def transferEvent : Nat := 6080
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 6078 .coefficient) (.value (.predecessor 1 6079 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6078 .coefficient)
      LeftAuthority6076.bound (LeftAuthority6076.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6077RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6076.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6079 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority6076.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6076.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6076.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound6080

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
