import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard704
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard732

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound106961
def owner : Owner := ⟨.program ⟨214⟩, ⟨26525⟩⟩
def transferEvent : Nat := 106961
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 106959 .coefficient, .predecessor 1 106960 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106959 .coefficient)
      LeftBound106814.bound (LeftBound106814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106960 .coefficient)
      LeftBound106797.bound (LeftBound106797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106814.bound, LeftBound106797.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106814.bound, LeftBound106797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106814.actual selector witness, LeftBound106797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106961

namespace LeftBound106964
def owner : Owner := ⟨.program ⟨214⟩, ⟨26525⟩⟩
def transferEvent : Nat := 106964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 106958 .summary, .result 106804 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106958 .summary)
      LeftBound106816.bound (LeftBound106816.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20456⟩⟩) (rawTerms := some (Proof.Events417.exact106958RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106816.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106804 .summary)
      LeftBound106799.bound (LeftBound106799.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26524⟩⟩) (rawTerms := some (Proof.Events417.exact106804RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106799.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound106816.bound, LeftBound106799.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106816.bound, LeftBound106799.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound106816.actual selector witness, LeftBound106799.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound106964

namespace LeftBound106968
def owner : Owner := ⟨.program ⟨214⟩, ⟨26526⟩⟩
def transferEvent : Nat := 106968
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106966 .coefficient) (.predecessor 1 106967 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106966 .coefficient)
      LeftBound106961.bound (LeftBound106961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106967 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106961.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106961.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106961.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106968

namespace LeftBound106969
def owner : Owner := ⟨.program ⟨214⟩, ⟨26526⟩⟩
def transferEvent : Nat := 106969
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106969

namespace LeftBound106970
def owner : Owner := ⟨.program ⟨214⟩, ⟨26526⟩⟩
def transferEvent : Nat := 106970
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 106965 .summary) (.transfer 106969) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106965 .summary)
      LeftBound106964.bound (LeftBound106964.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26525⟩⟩) (rawTerms := some (Proof.Events417.exact106965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound106964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106969)
      LeftBound106969.bound (LeftBound106969.actual selector witness) := by
  exact .transfer (LeftBound106969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound106964.bound LeftBound106969.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound106964.bound, LeftBound106969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound106964.actual selector witness) * (LeftBound106969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106970

namespace LeftBound106985
def owner : Owner := ⟨.program ⟨214⟩, ⟨26321⟩⟩
def transferEvent : Nat := 106985
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 106983 .coefficient) (.predecessor 1 106984 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106983 .coefficient)
      LeftBound101998.bound (LeftBound101998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact102002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106984 .coefficient)
      LeftAuthority106981.bound (LeftAuthority106981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106981.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106981.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101998.bound LeftAuthority106981.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101998.bound, LeftAuthority106981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101998.actual selector witness) * (LeftAuthority106981.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106985

namespace LeftBound106986
def owner : Owner := ⟨.program ⟨214⟩, ⟨26321⟩⟩
def transferEvent : Nat := 106986
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26319⟩⟩]⟩ [⟨.result 106982 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106982 .coefficient)
      LeftAuthority106981.bound (LeftAuthority106981.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26319⟩⟩) (rawTerms := some (Proof.Events417.exact106982RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106981.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106981.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106981.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106981.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound106986

namespace LeftBound106987
def owner : Owner := ⟨.program ⟨214⟩, ⟨26321⟩⟩
def transferEvent : Nat := 106987
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 102002 .summary) (.transfer 106986) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 102002 .summary)
      LeftBound102001.bound (LeftBound102001.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24900⟩⟩) (rawTerms := some (Proof.Events398.exact102002RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound102001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 106986)
      LeftBound106986.bound (LeftBound106986.actual selector witness) := by
  exact .transfer (LeftBound106986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound102001.bound LeftBound106986.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound102001.bound, LeftBound106986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound102001.actual selector witness) * (LeftBound106986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound106987

namespace LeftBound106998
def owner : Owner := ⟨.program ⟨214⟩, ⟨20311⟩⟩
def transferEvent : Nat := 106998
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 106996 .coefficient) (.value (.predecessor 1 106997 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 106996 .coefficient)
      LeftAuthority106994.bound (LeftAuthority106994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 106997 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority106994.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106994.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106994.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound106998

namespace LeftBound107002
def owner : Owner := ⟨.program ⟨214⟩, ⟨20312⟩⟩
def transferEvent : Nat := 107002
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 107000 .coefficient) (.predecessor 1 107001 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107000 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107001 .coefficient)
      LeftBound106998.bound (LeftBound106998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events417.exact106999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound106998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound106998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound106998.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound106998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound106998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107002

namespace LeftBound107003
def owner : Owner := ⟨.program ⟨214⟩, ⟨20312⟩⟩
def transferEvent : Nat := 107003
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20309⟩⟩]⟩ [⟨.result 106995 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 106995 .coefficient)
      LeftAuthority106994.bound (LeftAuthority106994.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20309⟩⟩) (rawTerms := some (Proof.Events417.exact106995RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority106994.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority106994.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority106994.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority106994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority106994.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound107003

namespace LeftBound107004
def owner : Owner := ⟨.program ⟨214⟩, ⟨20312⟩⟩
def transferEvent : Nat := 107004
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 107003) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 107003)
      LeftBound107003.bound (LeftBound107003.actual selector witness) := by
  exact .transfer (LeftBound107003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound107003.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound107003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound107003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107004

namespace LeftBound107075
def owner : Owner := ⟨.program ⟨214⟩, ⟨14783⟩⟩
def transferEvent : Nat := 107075
def frameStart : Nat := 107048
def rule : BoundRule := .identity (.predecessor 0 107074 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107074 .coefficient)
      LeftAuthority107072.bound (LeftAuthority107072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107072.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107072.derived selector witness)

def rawBound : CoeffClass := LeftAuthority107072.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority107072.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound107075

namespace LeftBound107092
def owner : Owner := ⟨.program ⟨214⟩, ⟨14824⟩⟩
def transferEvent : Nat := 107092
def frameStart : Nat := 107048
def rule : BoundRule := .sum [.predecessor 0 107090 .coefficient, .predecessor 1 107091 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107090 .coefficient)
      LeftBound107075.bound (LeftBound107075.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound107075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107091 .coefficient)
      LeftAuthority107088.bound (LeftAuthority107088.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority107088.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound107075.bound, LeftAuthority107088.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107075.bound, LeftAuthority107088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound107075.actual selector witness, LeftAuthority107088.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound107092

namespace LeftBound107095
def owner : Owner := ⟨.program ⟨214⟩, ⟨14825⟩⟩
def transferEvent : Nat := 107095
def frameStart : Nat := 107048
def rule : BoundRule := .identity (.predecessor 0 107094 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107094 .coefficient)
      LeftBound107092.bound (LeftBound107092.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound107092.derived selector witness)

def rawBound : CoeffClass := LeftBound107092.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound107092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound107092.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound107095

namespace LeftBound107101
def owner : Owner := ⟨.program ⟨214⟩, ⟨14826⟩⟩
def transferEvent : Nat := 107101
def frameStart : Nat := 107048
def rule : BoundRule := .product (.predecessor 0 107099 .coefficient) (.predecessor 1 107100 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 107099 .coefficient)
      LeftAuthority107097.bound (LeftAuthority107097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority107097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority107097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 107100 .coefficient)
      LeftBound107095.bound (LeftBound107095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events418.exact107096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound107095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound107095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority107097.bound LeftBound107095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority107097.bound, LeftBound107095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority107097.actual selector witness) * (LeftBound107095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound107101

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
