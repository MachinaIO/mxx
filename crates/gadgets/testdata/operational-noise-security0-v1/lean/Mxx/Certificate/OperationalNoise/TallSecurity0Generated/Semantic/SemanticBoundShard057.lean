import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard028
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard056

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound10008
def owner : Owner := ⟨.program ⟨214⟩, ⟨7862⟩⟩
def transferEvent : Nat := 10008
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 10006 .coefficient) (.value (.predecessor 1 10007 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10006 .coefficient)
      LeftAuthority10004.bound (LeftAuthority10004.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10005RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10004.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10007 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority10004.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10004.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10004.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10008

namespace LeftBound10011
def owner : Owner := ⟨.program ⟨214⟩, ⟨77⟩⟩
def transferEvent : Nat := 10011
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 10010 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10010 .coefficient)
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
end LeftBound10011

namespace LeftBound10015
def owner : Owner := ⟨.program ⟨214⟩, ⟨9631⟩⟩
def transferEvent : Nat := 10015
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 10013 .coefficient) (.predecessor 1 10014 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10013 .coefficient)
      LeftAuthority214.bound (LeftAuthority214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact215RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority214.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority214.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10014 .coefficient)
      LeftBound6447.bound (LeftBound6447.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6449RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6447.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6447.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority214.bound LeftBound6447.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority214.bound, LeftBound6447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority214.actual selector witness) * (LeftBound6447.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound10015

namespace LeftBound10019
def owner : Owner := ⟨.program ⟨214⟩, ⟨6763⟩⟩
def transferEvent : Nat := 10019
def frameStart : Nat := 0
def rule : BoundRule := .identity (.predecessor 0 10018 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10018 .coefficient)
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
end LeftBound10019

namespace LeftBound10023
def owner : Owner := ⟨.program ⟨214⟩, ⟨7371⟩⟩
def transferEvent : Nat := 10023
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10021 .coefficient) (.predecessor 1 10022 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10021 .coefficient)
      LeftBound6313.bound (LeftBound6313.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6313.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10022 .coefficient)
      LeftBound10019.bound (LeftBound10019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10019.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6313.bound LeftBound10019.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6313.bound, LeftBound10019.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6313.actual selector witness) * (LeftBound10019.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10023

namespace LeftBound10028
def owner : Owner := ⟨.program ⟨214⟩, ⟨9632⟩⟩
def transferEvent : Nat := 10028
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10026 .coefficient, .predecessor 1 10027 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10026 .coefficient)
      LeftBound10023.bound (LeftBound10023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10025RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10027 .coefficient)
      LeftBound10015.bound (LeftBound10015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10023.bound, LeftBound10015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10023.bound, LeftBound10015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10023.actual selector witness, LeftBound10015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10028

namespace LeftBound10032
def owner : Owner := ⟨.program ⟨214⟩, ⟨9633⟩⟩
def transferEvent : Nat := 10032
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10030 .coefficient, .predecessor 1 10031 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10030 .coefficient)
      LeftBound10028.bound (LeftBound10028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10028.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10031 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10028.bound, LeftBound10011.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10028.bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10028.actual selector witness, LeftBound10011.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10032

namespace LeftBound10033
def owner : Owner := ⟨.program ⟨214⟩, ⟨9633⟩⟩
def transferEvent : Nat := 10033
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨77⟩⟩]⟩ [⟨.result 10012 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10012 .coefficient)
      LeftBound10011.bound (LeftBound10011.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨77⟩⟩) (rawTerms := some (Proof.Events039.exact10012RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10011.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10011.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10011.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10033

namespace LeftBound10038
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def transferEvent : Nat := 10038
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10036 .coefficient) (.predecessor 1 10037 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10036 .coefficient)
      LeftBound10032.bound (LeftBound10032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10037 .coefficient)
      LeftBound10008.bound (LeftBound10008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10008.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10032.bound LeftBound10008.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10032.bound, LeftBound10008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10032.actual selector witness) * (LeftBound10008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10038

namespace LeftBound10039
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def transferEvent : Nat := 10039
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7861⟩⟩]⟩ [⟨.result 10005 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10005 .coefficient)
      LeftAuthority10004.bound (LeftAuthority10004.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7861⟩⟩) (rawTerms := some (Proof.Events039.exact10005RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10004.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10004.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10004.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10004.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10004.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10039

namespace LeftBound10040
def owner : Owner := ⟨.program ⟨214⟩, ⟨9634⟩⟩
def transferEvent : Nat := 10040
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10035 .summary) (.transfer 10039) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10035 .summary)
      LeftBound10033.bound (LeftBound10033.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9633⟩⟩) (rawTerms := some (Proof.Events039.exact10035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10039)
      LeftBound10039.bound (LeftBound10039.actual selector witness) := by
  exact .transfer (LeftBound10039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10033.bound LeftBound10039.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10033.bound, LeftBound10039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10033.actual selector witness) * (LeftBound10039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10040

namespace LeftBound10048
def owner : Owner := ⟨.program ⟨214⟩, ⟨11800⟩⟩
def transferEvent : Nat := 10048
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 10046 .coefficient, .predecessor 1 10047 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10046 .coefficient)
      LeftBound10038.bound (LeftBound10038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10038.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10047 .coefficient)
      LeftBound9997.bound (LeftBound9997.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9997.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9997.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10038.bound, LeftBound9997.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10038.bound, LeftBound9997.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10038.actual selector witness, LeftBound9997.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10048

namespace LeftBound10050
def owner : Owner := ⟨.program ⟨214⟩, ⟨11800⟩⟩
def transferEvent : Nat := 10050
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 10045 .summary, .result 10002 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10045 .summary)
      LeftBound10040.bound (LeftBound10040.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9634⟩⟩) (rawTerms := some (Proof.Events039.exact10045RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10002 .summary)
      LeftBound9999.bound (LeftBound9999.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11799⟩⟩) (rawTerms := some (Proof.Events039.exact10002RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9999.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound10040.bound, LeftBound9999.bound]
def bound : CoeffClass := .finite ⟨95445376, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10040.bound, LeftBound9999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound10040.actual selector witness, LeftBound9999.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound10050

namespace LeftBound10054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def transferEvent : Nat := 10054
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 10052 .coefficient) (.predecessor 1 10053 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 10052 .coefficient)
      LeftBound10048.bound (LeftBound10048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events039.exact10051RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 10053 .coefficient)
      LeftAuthority9967.bound (LeftAuthority9967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10048.bound LeftAuthority9967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10048.bound, LeftAuthority9967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10048.actual selector witness) * (LeftAuthority9967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10054

namespace LeftBound10055
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def transferEvent : Nat := 10055
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25162⟩⟩]⟩ [⟨.result 9968 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9968 .coefficient)
      LeftAuthority9967.bound (LeftAuthority9967.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25162⟩⟩) (rawTerms := some (Proof.Events038.exact9968RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority9967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority9967.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority9967.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority9967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority9967.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound10055

namespace LeftBound10056
def owner : Owner := ⟨.program ⟨214⟩, ⟨25163⟩⟩
def transferEvent : Nat := 10056
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10051 .summary) (.transfer 10055) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10051 .summary)
      LeftBound10050.bound (LeftBound10050.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11800⟩⟩) (rawTerms := some (Proof.Events039.exact10051RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10050.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 10055)
      LeftBound10055.bound (LeftBound10055.actual selector witness) := by
  exact .transfer (LeftBound10055.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10050.bound LeftBound10055.bound
def bound : CoeffClass := .finite ⟨350286057046016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10050.bound, LeftBound10055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10050.actual selector witness) * (LeftBound10055.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound10056

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
