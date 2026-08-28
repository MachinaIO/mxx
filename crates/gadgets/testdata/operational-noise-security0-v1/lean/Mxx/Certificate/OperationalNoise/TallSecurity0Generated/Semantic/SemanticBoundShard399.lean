import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard397
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard398

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound58631
def owner : Owner := ⟨.program ⟨214⟩, ⟨14959⟩⟩
def transferEvent : Nat := 58631
def frameStart : Nat := 58529
def rule : BoundRule := .product (.predecessor 0 58629 .coefficient) (.predecessor 1 58630 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58629 .coefficient)
      LeftAuthority58584.bound (LeftAuthority58584.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58585RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58584.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58584.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58630 .coefficient)
      LeftAuthority58627.bound (LeftAuthority58627.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58628RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58627.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58627.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority58584.bound LeftAuthority58627.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58584.bound, LeftAuthority58627.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority58584.actual selector witness) * (LeftAuthority58627.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58631

namespace LeftBound58639
def owner : Owner := ⟨.program ⟨214⟩, ⟨14960⟩⟩
def transferEvent : Nat := 58639
def frameStart : Nat := 58529
def rule : BoundRule := .sum [.predecessor 0 58637 .coefficient, .predecessor 1 58638 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58637 .coefficient)
      LeftAuthority58635.bound (LeftAuthority58635.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58635.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58638 .coefficient)
      LeftBound58631.bound (LeftBound58631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority58635.bound, LeftBound58631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58635.bound, LeftBound58631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority58635.actual selector witness, LeftBound58631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58639

namespace LeftBound58643
def owner : Owner := ⟨.program ⟨214⟩, ⟨24997⟩⟩
def transferEvent : Nat := 58643
def frameStart : Nat := 58529
def rule : BoundRule := .sum [.predecessor 0 58641 .coefficient, .predecessor 1 58642 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58641 .coefficient)
      LeftBound58639.bound (LeftBound58639.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58640RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58639.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58639.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58642 .coefficient)
      LeftBound58620.bound (LeftBound58620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58639.bound, LeftBound58620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58639.bound, LeftBound58620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58639.actual selector witness, LeftBound58620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58643

namespace LeftBound58656
def owner : Owner := ⟨.program ⟨214⟩, ⟨24995⟩⟩
def transferEvent : Nat := 58656
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 58654 .coefficient, .predecessor 1 58655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58654 .coefficient)
      LeftBound58477.bound (LeftBound58477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58655 .coefficient)
      LeftBound58460.bound (LeftBound58460.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58460.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58460.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58477.bound, LeftBound58460.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58477.bound, LeftBound58460.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58477.actual selector witness, LeftBound58460.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58656

namespace LeftBound58659
def owner : Owner := ⟨.program ⟨214⟩, ⟨24995⟩⟩
def transferEvent : Nat := 58659
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 58653 .summary, .result 58467 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58653 .summary)
      LeftBound58479.bound (LeftBound58479.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19103⟩⟩) (rawTerms := some (Proof.Events229.exact58653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58467 .summary)
      LeftBound58462.bound (LeftBound58462.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24994⟩⟩) (rawTerms := some (Proof.Events228.exact58467RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58462.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58479.bound, LeftBound58462.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58479.bound, LeftBound58462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58479.actual selector witness, LeftBound58462.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58659

namespace LeftBound58663
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def transferEvent : Nat := 58663
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58661 .coefficient) (.predecessor 1 58662 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58661 .coefficient)
      LeftBound58656.bound (LeftBound58656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58662 .coefficient)
      LeftAuthority58382.bound (LeftAuthority58382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events228.exact58383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58382.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58656.bound LeftAuthority58382.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58656.bound, LeftAuthority58382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58656.actual selector witness) * (LeftAuthority58382.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58663

namespace LeftBound58664
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def transferEvent : Nat := 58664
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26577⟩⟩]⟩ [⟨.result 58383 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58383 .coefficient)
      LeftAuthority58382.bound (LeftAuthority58382.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26577⟩⟩) (rawTerms := some (Proof.Events228.exact58383RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58382.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority58382.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58382.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58664

namespace LeftBound58665
def owner : Owner := ⟨.program ⟨214⟩, ⟨26579⟩⟩
def transferEvent : Nat := 58665
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 58660 .summary) (.transfer 58664) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58660 .summary)
      LeftBound58659.bound (LeftBound58659.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24995⟩⟩) (rawTerms := some (Proof.Events229.exact58660RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound58659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58664)
      LeftBound58664.bound (LeftBound58664.actual selector witness) := by
  exact .transfer (LeftBound58664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound58659.bound LeftBound58664.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58659.bound, LeftBound58664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound58659.actual selector witness) * (LeftBound58664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58665

namespace LeftBound58676
def owner : Owner := ⟨.program ⟨214⟩, ⟨20542⟩⟩
def transferEvent : Nat := 58676
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 58674 .coefficient) (.value (.predecessor 1 58675 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58674 .coefficient)
      LeftAuthority58672.bound (LeftAuthority58672.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58675 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority58672.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58672.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58672.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound58676

namespace LeftBound58680
def owner : Owner := ⟨.program ⟨214⟩, ⟨20543⟩⟩
def transferEvent : Nat := 58680
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 58678 .coefficient) (.predecessor 1 58679 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58678 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58679 .coefficient)
      LeftBound58676.bound (LeftBound58676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound58676.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound58676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound58676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58680

namespace LeftBound58681
def owner : Owner := ⟨.program ⟨214⟩, ⟨20543⟩⟩
def transferEvent : Nat := 58681
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20540⟩⟩]⟩ [⟨.result 58673 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 58673 .coefficient)
      LeftAuthority58672.bound (LeftAuthority58672.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20540⟩⟩) (rawTerms := some (Proof.Events229.exact58673RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58672.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority58672.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58672.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority58672.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound58681

namespace LeftBound58682
def owner : Owner := ⟨.program ⟨214⟩, ⟨20543⟩⟩
def transferEvent : Nat := 58682
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 58681) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 58681)
      LeftBound58681.bound (LeftBound58681.actual selector witness) := by
  exact .transfer (LeftBound58681.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound58681.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound58681.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound58681.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58682

namespace LeftBound58777
def owner : Owner := ⟨.program ⟨214⟩, ⟨14958⟩⟩
def transferEvent : Nat := 58777
def frameStart : Nat := 58738
def rule : BoundRule := .identity (.predecessor 0 58776 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58776 .coefficient)
      LeftAuthority58774.bound (LeftAuthority58774.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58774.derived selector witness)

def rawBound : CoeffClass := LeftAuthority58774.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority58774.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound58777

namespace LeftBound58794
def owner : Owner := ⟨.program ⟨214⟩, ⟨14997⟩⟩
def transferEvent : Nat := 58794
def frameStart : Nat := 58738
def rule : BoundRule := .sum [.predecessor 0 58792 .coefficient, .predecessor 1 58793 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58792 .coefficient)
      LeftBound58777.bound (LeftBound58777.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound58777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58793 .coefficient)
      LeftAuthority58790.bound (LeftAuthority58790.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority58790.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound58777.bound, LeftAuthority58790.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58777.bound, LeftAuthority58790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound58777.actual selector witness, LeftAuthority58790.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound58794

namespace LeftBound58797
def owner : Owner := ⟨.program ⟨214⟩, ⟨14998⟩⟩
def transferEvent : Nat := 58797
def frameStart : Nat := 58738
def rule : BoundRule := .identity (.predecessor 0 58796 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58796 .coefficient)
      LeftBound58794.bound (LeftBound58794.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound58794.derived selector witness)

def rawBound : CoeffClass := LeftBound58794.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound58794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound58794.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound58797

namespace LeftBound58803
def owner : Owner := ⟨.program ⟨214⟩, ⟨14999⟩⟩
def transferEvent : Nat := 58803
def frameStart : Nat := 58738
def rule : BoundRule := .product (.predecessor 0 58801 .coefficient) (.predecessor 1 58802 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 58801 .coefficient)
      LeftAuthority58799.bound (LeftAuthority58799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority58799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority58799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 58802 .coefficient)
      LeftBound58797.bound (LeftBound58797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events229.exact58798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound58797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound58797.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority58799.bound LeftBound58797.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority58799.bound, LeftBound58797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority58799.actual selector witness) * (LeftBound58797.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound58803

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
