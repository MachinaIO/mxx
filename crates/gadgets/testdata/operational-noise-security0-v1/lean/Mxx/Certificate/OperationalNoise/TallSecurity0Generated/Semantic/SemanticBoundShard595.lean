import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard594

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86920
def owner : Owner := ⟨.program ⟨214⟩, ⟨27000⟩⟩
def transferEvent : Nat := 86920
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26998⟩⟩]⟩ [⟨.result 86641 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86641 .coefficient)
      LeftAuthority86640.bound (LeftAuthority86640.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26998⟩⟩) (rawTerms := some (Proof.Events338.exact86641RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86640.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86640.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86640.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86640.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86920

namespace LeftBound86921
def owner : Owner := ⟨.program ⟨214⟩, ⟨27000⟩⟩
def transferEvent : Nat := 86921
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86916 .summary) (.transfer 86920) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86916 .summary)
      LeftBound86915.bound (LeftBound86915.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25298⟩⟩) (rawTerms := some (Proof.Events339.exact86916RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86920)
      LeftBound86920.bound (LeftBound86920.actual selector witness) := by
  exact .transfer (LeftBound86920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86915.bound LeftBound86920.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86915.bound, LeftBound86920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86915.actual selector witness) * (LeftBound86920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86921

namespace LeftBound86932
def owner : Owner := ⟨.program ⟨214⟩, ⟨20826⟩⟩
def transferEvent : Nat := 86932
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 86930 .coefficient) (.value (.predecessor 1 86931 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86930 .coefficient)
      LeftAuthority86928.bound (LeftAuthority86928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86928.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86928.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86931 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority86928.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86928.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86928.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86932

namespace LeftBound86936
def owner : Owner := ⟨.program ⟨214⟩, ⟨20827⟩⟩
def transferEvent : Nat := 86936
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86934 .coefficient) (.predecessor 1 86935 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86934 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86935 .coefficient)
      LeftBound86932.bound (LeftBound86932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86932.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86932.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound86932.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound86932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound86932.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86936

namespace LeftBound86937
def owner : Owner := ⟨.program ⟨214⟩, ⟨20827⟩⟩
def transferEvent : Nat := 86937
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20824⟩⟩]⟩ [⟨.result 86929 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86929 .coefficient)
      LeftAuthority86928.bound (LeftAuthority86928.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20824⟩⟩) (rawTerms := some (Proof.Events339.exact86929RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86928.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86928.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86928.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86928.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86937

namespace LeftBound86938
def owner : Owner := ⟨.program ⟨214⟩, ⟨20827⟩⟩
def transferEvent : Nat := 86938
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 86937) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86937)
      LeftBound86937.bound (LeftBound86937.actual selector witness) := by
  exact .transfer (LeftBound86937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound86937.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound86937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound86937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86938

namespace LeftBound87033
def owner : Owner := ⟨.program ⟨214⟩, ⟨15423⟩⟩
def transferEvent : Nat := 87033
def frameStart : Nat := 86994
def rule : BoundRule := .identity (.predecessor 0 87032 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87032 .coefficient)
      LeftAuthority87030.bound (LeftAuthority87030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact87031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87030.derived selector witness)

def rawBound : CoeffClass := LeftAuthority87030.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority87030.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87033

namespace LeftBound87050
def owner : Owner := ⟨.program ⟨214⟩, ⟨15462⟩⟩
def transferEvent : Nat := 87050
def frameStart : Nat := 86994
def rule : BoundRule := .sum [.predecessor 0 87048 .coefficient, .predecessor 1 87049 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87048 .coefficient)
      LeftBound87033.bound (LeftBound87033.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87033.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87049 .coefficient)
      LeftAuthority87046.bound (LeftAuthority87046.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority87046.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87033.bound, LeftAuthority87046.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87033.bound, LeftAuthority87046.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87033.actual selector witness, LeftAuthority87046.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87050

namespace LeftBound87053
def owner : Owner := ⟨.program ⟨214⟩, ⟨15463⟩⟩
def transferEvent : Nat := 87053
def frameStart : Nat := 86994
def rule : BoundRule := .identity (.predecessor 0 87052 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87052 .coefficient)
      LeftBound87050.bound (LeftBound87050.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound87050.derived selector witness)

def rawBound : CoeffClass := LeftBound87050.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87050.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound87050.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound87053

namespace LeftBound87059
def owner : Owner := ⟨.program ⟨214⟩, ⟨15464⟩⟩
def transferEvent : Nat := 87059
def frameStart : Nat := 86994
def rule : BoundRule := .product (.predecessor 0 87057 .coefficient) (.predecessor 1 87058 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87057 .coefficient)
      LeftAuthority87055.bound (LeftAuthority87055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87055.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87058 .coefficient)
      LeftBound87053.bound (LeftBound87053.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87054RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87053.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87053.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority87055.bound LeftBound87053.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87055.bound, LeftBound87053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority87055.actual selector witness) * (LeftBound87053.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87059

namespace LeftBound87067
def owner : Owner := ⟨.program ⟨214⟩, ⟨15465⟩⟩
def transferEvent : Nat := 87067
def frameStart : Nat := 86994
def rule : BoundRule := .sum [.predecessor 0 87065 .coefficient, .predecessor 1 87066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87065 .coefficient)
      LeftAuthority87063.bound (LeftAuthority87063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87066 .coefficient)
      LeftBound87059.bound (LeftBound87059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87063.bound, LeftBound87059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87063.bound, LeftBound87059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87063.actual selector witness, LeftBound87059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87067

namespace LeftBound87071
def owner : Owner := ⟨.program ⟨214⟩, ⟨26999⟩⟩
def transferEvent : Nat := 87071
def frameStart : Nat := 86994
def rule : BoundRule := .product (.predecessor 0 87069 .coefficient) (.predecessor 1 87070 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87069 .coefficient)
      LeftBound87067.bound (LeftBound87067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87070 .coefficient)
      LeftAuthority87044.bound (LeftAuthority87044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87044.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87044.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound87067.bound LeftAuthority87044.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87067.bound, LeftAuthority87044.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound87067.actual selector witness) * (LeftAuthority87044.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87071

namespace LeftBound87082
def owner : Owner := ⟨.program ⟨214⟩, ⟨17334⟩⟩
def transferEvent : Nat := 87082
def frameStart : Nat := 86994
def rule : BoundRule := .product (.predecessor 0 87080 .coefficient) (.predecessor 1 87081 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87080 .coefficient)
      LeftAuthority87055.bound (LeftAuthority87055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87055.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87081 .coefficient)
      LeftAuthority87078.bound (LeftAuthority87078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87078.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority87055.bound LeftAuthority87078.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87055.bound, LeftAuthority87078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority87055.actual selector witness) * (LeftAuthority87078.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound87082

namespace LeftBound87090
def owner : Owner := ⟨.program ⟨214⟩, ⟨17335⟩⟩
def transferEvent : Nat := 87090
def frameStart : Nat := 86994
def rule : BoundRule := .sum [.predecessor 0 87088 .coefficient, .predecessor 1 87089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87088 .coefficient)
      LeftAuthority87086.bound (LeftAuthority87086.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority87086.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority87086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87089 .coefficient)
      LeftBound87082.bound (LeftBound87082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87082.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87082.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority87086.bound, LeftBound87082.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority87086.bound, LeftBound87082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority87086.actual selector witness, LeftBound87082.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87090

namespace LeftBound87094
def owner : Owner := ⟨.program ⟨214⟩, ⟨27003⟩⟩
def transferEvent : Nat := 87094
def frameStart : Nat := 86994
def rule : BoundRule := .sum [.predecessor 0 87092 .coefficient, .predecessor 1 87093 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87092 .coefficient)
      LeftBound87090.bound (LeftBound87090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87093 .coefficient)
      LeftBound87071.bound (LeftBound87071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound87071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound87071.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound87090.bound, LeftBound87071.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound87090.bound, LeftBound87071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound87090.actual selector witness, LeftBound87071.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87094

namespace LeftBound87107
def owner : Owner := ⟨.program ⟨214⟩, ⟨27001⟩⟩
def transferEvent : Nat := 87107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 87105 .coefficient, .predecessor 1 87106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 87105 .coefficient)
      LeftBound86936.bound (LeftBound86936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events340.exact87104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86936.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86936.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 87106 .coefficient)
      LeftBound86919.bound (LeftBound86919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events339.exact86926RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86919.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86936.bound, LeftBound86919.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86936.bound, LeftBound86919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86936.actual selector witness, LeftBound86919.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound87107

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
