import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard029
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound79951
def owner : Owner := ⟨.program ⟨214⟩, ⟨10346⟩⟩
def transferEvent : Nat := 79951
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 79949 .coefficient) (.predecessor 1 79950 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79949 .coefficient)
      LeftAuthority3830.bound (LeftAuthority3830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events014.exact3831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3830.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79950 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3830.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3830.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3830.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound79951

namespace LeftBound79956
def owner : Owner := ⟨.program ⟨214⟩, ⟨7226⟩⟩
def transferEvent : Nat := 79956
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79954 .coefficient) (.predecessor 1 79955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79954 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79955 .coefficient)
      LeftBound6497.bound (LeftBound6497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6497.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6497.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound6497.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound6497.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound6497.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79956

namespace LeftBound79961
def owner : Owner := ⟨.program ⟨214⟩, ⟨10347⟩⟩
def transferEvent : Nat := 79961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79959 .coefficient, .predecessor 1 79960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79959 .coefficient)
      LeftBound79956.bound (LeftBound79956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79960 .coefficient)
      LeftBound79951.bound (LeftBound79951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79951.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79956.bound, LeftBound79951.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79956.bound, LeftBound79951.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79956.actual selector witness, LeftBound79951.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79961

namespace LeftBound79965
def owner : Owner := ⟨.program ⟨214⟩, ⟨10348⟩⟩
def transferEvent : Nat := 79965
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79963 .coefficient, .predecessor 1 79964 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79963 .coefficient)
      LeftBound79961.bound (LeftBound79961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79964 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79961.bound, LeftBound6489.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79961.bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79961.actual selector witness, LeftBound6489.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79965

namespace LeftBound79966
def owner : Owner := ⟨.program ⟨214⟩, ⟨10348⟩⟩
def transferEvent : Nat := 79966
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨84⟩⟩]⟩ [⟨.result 6490 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6490 .coefficient)
      LeftBound6489.bound (LeftBound6489.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨84⟩⟩) (rawTerms := some (Proof.Events025.exact6490RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6489.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6489.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound6489.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6489.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound6489.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79966

namespace LeftBound79971
def owner : Owner := ⟨.program ⟨214⟩, ⟨10349⟩⟩
def transferEvent : Nat := 79971
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79969 .coefficient) (.predecessor 1 79970 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79969 .coefficient)
      LeftBound79965.bound (LeftBound79965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79965.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79970 .coefficient)
      LeftBound6486.bound (LeftBound6486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79965.bound LeftBound6486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79965.bound, LeftBound6486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79965.actual selector witness) * (LeftBound6486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79971

namespace LeftBound79972
def owner : Owner := ⟨.program ⟨214⟩, ⟨10349⟩⟩
def transferEvent : Nat := 79972
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7882⟩⟩]⟩ [⟨.result 6483 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6483 .coefficient)
      LeftAuthority6482.bound (LeftAuthority6482.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7882⟩⟩) (rawTerms := some (Proof.Events025.exact6483RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6482.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6482.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6482.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79972

namespace LeftBound79973
def owner : Owner := ⟨.program ⟨214⟩, ⟨10349⟩⟩
def transferEvent : Nat := 79973
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79968 .summary) (.transfer 79972) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79968 .summary)
      LeftBound79966.bound (LeftBound79966.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10348⟩⟩) (rawTerms := some (Proof.Events312.exact79968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79972)
      LeftBound79972.bound (LeftBound79972.actual selector witness) := by
  exact .transfer (LeftBound79972.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79966.bound LeftBound79972.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79966.bound, LeftBound79972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79966.actual selector witness) * (LeftBound79972.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79973

namespace LeftBound79981
def owner : Owner := ⟨.program ⟨214⟩, ⟨13357⟩⟩
def transferEvent : Nat := 79981
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 79979 .coefficient, .predecessor 1 79980 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79979 .coefficient)
      LeftBound79971.bound (LeftBound79971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79980 .coefficient)
      LeftBound79943.bound (LeftBound79943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79971.bound, LeftBound79943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79971.bound, LeftBound79943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79971.actual selector witness, LeftBound79943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79981

namespace LeftBound79983
def owner : Owner := ⟨.program ⟨214⟩, ⟨13357⟩⟩
def transferEvent : Nat := 79983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 79978 .summary, .result 79948 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79978 .summary)
      LeftBound79973.bound (LeftBound79973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10349⟩⟩) (rawTerms := some (Proof.Events312.exact79978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79948 .summary)
      LeftBound79945.bound (LeftBound79945.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13356⟩⟩) (rawTerms := some (Proof.Events312.exact79948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79945.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound79973.bound, LeftBound79945.bound]
def bound : CoeffClass := .finite ⟨95470336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79973.bound, LeftBound79945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound79973.actual selector witness, LeftBound79945.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound79983

namespace LeftBound79987
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def transferEvent : Nat := 79987
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 79985 .coefficient) (.predecessor 1 79986 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79985 .coefficient)
      LeftBound79981.bound (LeftBound79981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79981.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79986 .coefficient)
      LeftAuthority79914.bound (LeftAuthority79914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79981.bound LeftAuthority79914.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79981.bound, LeftAuthority79914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79981.actual selector witness) * (LeftAuthority79914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79987

namespace LeftBound79988
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def transferEvent : Nat := 79988
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25758⟩⟩]⟩ [⟨.result 79915 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79915 .coefficient)
      LeftAuthority79914.bound (LeftAuthority79914.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25758⟩⟩) (rawTerms := some (Proof.Events312.exact79915RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79914.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority79914.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79914.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound79988

namespace LeftBound79989
def owner : Owner := ⟨.program ⟨214⟩, ⟨25759⟩⟩
def transferEvent : Nat := 79989
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 79984 .summary) (.transfer 79988) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 79984 .summary)
      LeftBound79983.bound (LeftBound79983.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13357⟩⟩) (rawTerms := some (Proof.Events312.exact79984RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound79983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 79988)
      LeftBound79988.bound (LeftBound79988.actual selector witness) := by
  exact .transfer (LeftBound79988.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound79983.bound LeftBound79988.bound
def bound : CoeffClass := .finite ⟨350377660645376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79983.bound, LeftBound79988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound79983.actual selector witness) * (LeftBound79988.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound79989

namespace LeftBound80000
def owner : Owner := ⟨.program ⟨214⟩, ⟨20250⟩⟩
def transferEvent : Nat := 80000
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 79998 .coefficient) (.value (.predecessor 1 79999 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 79998 .coefficient)
      LeftAuthority79996.bound (LeftAuthority79996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority79996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority79996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 79999 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority79996.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority79996.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority79996.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound80000

namespace LeftBound80004
def owner : Owner := ⟨.program ⟨214⟩, ⟨5540⟩⟩
def transferEvent : Nat := 80004
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 80002 .coefficient) (.predecessor 1 80003 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80002 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80003 .coefficient)
      LeftAuthority6549.bound (LeftAuthority6549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftAuthority6549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftAuthority6549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftAuthority6549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 16) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound80004

namespace LeftBound80009
def owner : Owner := ⟨.program ⟨214⟩, ⟨5541⟩⟩
def transferEvent : Nat := 80009
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 80007 .coefficient, .predecessor 1 80008 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 80007 .coefficient)
      LeftBound80004.bound (LeftBound80004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80004.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 80008 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound80004.bound, LeftAuthority6547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80004.bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound80004.actual selector witness, LeftAuthority6547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound80009

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
