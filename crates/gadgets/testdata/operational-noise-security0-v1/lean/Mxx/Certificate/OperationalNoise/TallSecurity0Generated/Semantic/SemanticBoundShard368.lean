import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard061
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard367

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54554
def owner : Owner := ⟨.program ⟨214⟩, ⟨11644⟩⟩
def transferEvent : Nat := 54554
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54552 .coefficient, .predecessor 1 54553 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54552 .coefficient)
      LeftBound54550.bound (LeftBound54550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54551RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54550.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54553 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54550.bound, LeftBound10471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54550.bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54550.actual selector witness, LeftBound10471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54554

namespace LeftBound54555
def owner : Owner := ⟨.program ⟨214⟩, ⟨11644⟩⟩
def transferEvent : Nat := 54555
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨95⟩⟩]⟩ [⟨.result 10472 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10472 .coefficient)
      LeftBound10471.bound (LeftBound10471.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨95⟩⟩) (rawTerms := some (Proof.Events040.exact10472RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10471.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10471.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10471.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54555

namespace LeftBound54560
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def transferEvent : Nat := 54560
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54558 .coefficient) (.predecessor 1 54559 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54558 .coefficient)
      LeftBound54554.bound (LeftBound54554.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54554.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54559 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound54554.bound LeftAuthority2524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54554.bound, LeftAuthority2524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound54554.actual selector witness) * (LeftAuthority2524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54560

namespace LeftBound54561
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def transferEvent : Nat := 54561
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14650⟩⟩], []⟩ [⟨.result 2525 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2525 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14650⟩⟩) (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2524.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2524.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54561

namespace LeftBound54562
def owner : Owner := ⟨.program ⟨214⟩, ⟨14653⟩⟩
def transferEvent : Nat := 54562
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54557 .summary) (.transfer 54561) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54557 .summary)
      LeftBound54555.bound (LeftBound54555.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11644⟩⟩) (rawTerms := some (Proof.Events213.exact54557RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54555.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54561)
      LeftBound54561.bound (LeftBound54561.actual selector witness) := by
  exact .transfer (LeftBound54561.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound54555.bound LeftBound54561.bound
def bound : CoeffClass := .finite ⟨23296, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54555.bound, LeftBound54561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound54555.actual selector witness) * (LeftBound54561.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54562

namespace LeftBound54568
def owner : Owner := ⟨.program ⟨214⟩, ⟨14654⟩⟩
def transferEvent : Nat := 54568
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54566 .coefficient) (.predecessor 1 54567 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54566 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54567 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2524.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2524.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2524.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54568

namespace LeftBound54573
def owner : Owner := ⟨.program ⟨214⟩, ⟨7256⟩⟩
def transferEvent : Nat := 54573
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54571 .coefficient) (.predecessor 1 54572 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54571 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54572 .coefficient)
      LeftBound10520.bound (LeftBound10520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10520.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound10520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound10520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound10520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54573

namespace LeftBound54578
def owner : Owner := ⟨.program ⟨214⟩, ⟨14655⟩⟩
def transferEvent : Nat := 54578
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54576 .coefficient, .predecessor 1 54577 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54576 .coefficient)
      LeftBound54573.bound (LeftBound54573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54573.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54577 .coefficient)
      LeftBound54568.bound (LeftBound54568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54568.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54573.bound, LeftBound54568.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54573.bound, LeftBound54568.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54573.actual selector witness, LeftBound54568.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54578

namespace LeftBound54582
def owner : Owner := ⟨.program ⟨214⟩, ⟨14656⟩⟩
def transferEvent : Nat := 54582
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54580 .coefficient, .predecessor 1 54581 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54580 .coefficient)
      LeftBound54578.bound (LeftBound54578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54578.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54581 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54578.bound, LeftBound10512.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54578.bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54578.actual selector witness, LeftBound10512.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54582

namespace LeftBound54583
def owner : Owner := ⟨.program ⟨214⟩, ⟨14656⟩⟩
def transferEvent : Nat := 54583
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨76⟩⟩]⟩ [⟨.result 10513 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10513 .coefficient)
      LeftBound10512.bound (LeftBound10512.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨76⟩⟩) (rawTerms := some (Proof.Events041.exact10513RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10512.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10512.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10512.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54583

namespace LeftBound54588
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def transferEvent : Nat := 54588
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54586 .coefficient) (.predecessor 1 54587 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54586 .coefficient)
      LeftBound54582.bound (LeftBound54582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54587 .coefficient)
      LeftBound10509.bound (LeftBound10509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events041.exact10510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10509.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54582.bound LeftBound10509.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54582.bound, LeftBound10509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54582.actual selector witness) * (LeftBound10509.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54588

namespace LeftBound54589
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def transferEvent : Nat := 54589
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7858⟩⟩]⟩ [⟨.result 10506 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10506 .coefficient)
      LeftAuthority10505.bound (LeftAuthority10505.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7858⟩⟩) (rawTerms := some (Proof.Events041.exact10506RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority10505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority10505.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority10505.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority10505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority10505.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54589

namespace LeftBound54590
def owner : Owner := ⟨.program ⟨214⟩, ⟨14657⟩⟩
def transferEvent : Nat := 54590
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54585 .summary) (.transfer 54589) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54585 .summary)
      LeftBound54583.bound (LeftBound54583.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14656⟩⟩) (rawTerms := some (Proof.Events213.exact54585RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54589)
      LeftBound54589.bound (LeftBound54589.actual selector witness) := by
  exact .transfer (LeftBound54589.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54583.bound LeftBound54589.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54583.bound, LeftBound54589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54583.actual selector witness) * (LeftBound54589.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54590

namespace LeftBound54598
def owner : Owner := ⟨.program ⟨214⟩, ⟨14658⟩⟩
def transferEvent : Nat := 54598
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54596 .coefficient, .predecessor 1 54597 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54596 .coefficient)
      LeftBound54588.bound (LeftBound54588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54588.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54588.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54597 .coefficient)
      LeftBound54560.bound (LeftBound54560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54588.bound, LeftBound54560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54588.bound, LeftBound54560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54588.actual selector witness, LeftBound54560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54598

namespace LeftBound54600
def owner : Owner := ⟨.program ⟨214⟩, ⟨14658⟩⟩
def transferEvent : Nat := 54600
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54595 .summary, .result 54565 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54595 .summary)
      LeftBound54590.bound (LeftBound54590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14657⟩⟩) (rawTerms := some (Proof.Events213.exact54595RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54565 .summary)
      LeftBound54562.bound (LeftBound54562.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14653⟩⟩) (rawTerms := some (Proof.Events213.exact54565RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54562.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54590.bound, LeftBound54562.bound]
def bound : CoeffClass := .finite ⟨95443712, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54590.bound, LeftBound54562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54590.actual selector witness, LeftBound54562.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54600

namespace LeftBound54604
def owner : Owner := ⟨.program ⟨214⟩, ⟨26226⟩⟩
def transferEvent : Nat := 54604
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54602 .coefficient) (.predecessor 1 54603 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54602 .coefficient)
      LeftBound54598.bound (LeftBound54598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54603 .coefficient)
      LeftAuthority54536.bound (LeftAuthority54536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54536.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54598.bound LeftAuthority54536.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54598.bound, LeftAuthority54536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54598.actual selector witness) * (LeftAuthority54536.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54604

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
