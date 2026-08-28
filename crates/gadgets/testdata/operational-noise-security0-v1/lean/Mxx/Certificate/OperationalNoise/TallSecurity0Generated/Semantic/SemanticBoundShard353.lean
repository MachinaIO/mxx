import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard044
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard045
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard352

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound52560
def owner : Owner := ⟨.program ⟨214⟩, ⟨16683⟩⟩
def transferEvent : Nat := 52560
def frameStart : Nat := 52472
def rule : BoundRule := .product (.predecessor 0 52558 .coefficient) (.predecessor 1 52559 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52558 .coefficient)
      LeftAuthority52533.bound (LeftAuthority52533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52534RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52533.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52559 .coefficient)
      LeftAuthority52556.bound (LeftAuthority52556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52557RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52556.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52556.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority52533.bound LeftAuthority52556.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52533.bound, LeftAuthority52556.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority52533.actual selector witness) * (LeftAuthority52556.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52560

namespace LeftBound52568
def owner : Owner := ⟨.program ⟨214⟩, ⟨16684⟩⟩
def transferEvent : Nat := 52568
def frameStart : Nat := 52472
def rule : BoundRule := .sum [.predecessor 0 52566 .coefficient, .predecessor 1 52567 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52566 .coefficient)
      LeftAuthority52564.bound (LeftAuthority52564.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority52564.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority52564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52567 .coefficient)
      LeftBound52560.bound (LeftBound52560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52562RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority52564.bound, LeftBound52560.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority52564.bound, LeftBound52560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority52564.actual selector witness, LeftBound52560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52568

namespace LeftBound52572
def owner : Owner := ⟨.program ⟨214⟩, ⟨29403⟩⟩
def transferEvent : Nat := 52572
def frameStart : Nat := 52472
def rule : BoundRule := .sum [.predecessor 0 52570 .coefficient, .predecessor 1 52571 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52570 .coefficient)
      LeftBound52568.bound (LeftBound52568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52569RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52571 .coefficient)
      LeftBound52549.bound (LeftBound52549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52549.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52568.bound, LeftBound52549.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52568.bound, LeftBound52549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52568.actual selector witness, LeftBound52549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52572

namespace LeftBound52585
def owner : Owner := ⟨.program ⟨214⟩, ⟨29401⟩⟩
def transferEvent : Nat := 52585
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52583 .coefficient, .predecessor 1 52584 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52583 .coefficient)
      LeftBound52414.bound (LeftBound52414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52414.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52584 .coefficient)
      LeftBound52397.bound (LeftBound52397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events204.exact52404RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52414.bound, LeftBound52397.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52414.bound, LeftBound52397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52414.actual selector witness, LeftBound52397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52585

namespace LeftBound52588
def owner : Owner := ⟨.program ⟨214⟩, ⟨29401⟩⟩
def transferEvent : Nat := 52588
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 52582 .summary, .result 52404 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52582 .summary)
      LeftBound52416.bound (LeftBound52416.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22415⟩⟩) (rawTerms := some (Proof.Events205.exact52582RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52416.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52404 .summary)
      LeftBound52399.bound (LeftBound52399.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29400⟩⟩) (rawTerms := some (Proof.Events204.exact52404RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52399.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52416.bound, LeftBound52399.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52416.bound, LeftBound52399.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52416.actual selector witness, LeftBound52399.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52588

namespace LeftBound52612
def owner : Owner := ⟨.program ⟨214⟩, ⟨12577⟩⟩
def transferEvent : Nat := 52612
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 52610 .coefficient) (.predecessor 1 52611 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52610 .coefficient)
      LeftAuthority2429.bound (LeftAuthority2429.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2429.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2429.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52611 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2429.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2429.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2429.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52612

namespace LeftBound52617
def owner : Owner := ⟨.program ⟨214⟩, ⟨7280⟩⟩
def transferEvent : Nat := 52617
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52615 .coefficient) (.predecessor 1 52616 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52615 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52616 .coefficient)
      LeftBound8475.bound (LeftBound8475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8475.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound8475.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound8475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound8475.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52617

namespace LeftBound52622
def owner : Owner := ⟨.program ⟨214⟩, ⟨12578⟩⟩
def transferEvent : Nat := 52622
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52620 .coefficient, .predecessor 1 52621 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52620 .coefficient)
      LeftBound52617.bound (LeftBound52617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52621 .coefficient)
      LeftBound52612.bound (LeftBound52612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52617.bound, LeftBound52612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52617.bound, LeftBound52612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52617.actual selector witness, LeftBound52612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52622

namespace LeftBound52626
def owner : Owner := ⟨.program ⟨214⟩, ⟨12579⟩⟩
def transferEvent : Nat := 52626
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52624 .coefficient, .predecessor 1 52625 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52624 .coefficient)
      LeftBound52622.bound (LeftBound52622.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52623RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52622.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52625 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52622.bound, LeftBound8467.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52622.bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52622.actual selector witness, LeftBound8467.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52626

namespace LeftBound52627
def owner : Owner := ⟨.program ⟨214⟩, ⟨12579⟩⟩
def transferEvent : Nat := 52627
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨100⟩⟩]⟩ [⟨.result 8468 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8468 .coefficient)
      LeftBound8467.bound (LeftBound8467.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨100⟩⟩) (rawTerms := some (Proof.Events033.exact8468RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8467.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8467.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound8467.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound8467.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52627

namespace LeftBound52632
def owner : Owner := ⟨.program ⟨214⟩, ⟨12580⟩⟩
def transferEvent : Nat := 52632
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52630 .coefficient) (.predecessor 1 52631 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52630 .coefficient)
      LeftBound52626.bound (LeftBound52626.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52626.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52626.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52631 .coefficient)
      LeftAuthority2432.bound (LeftAuthority2432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2432.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound52626.bound LeftAuthority2432.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52626.bound, LeftAuthority2432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound52626.actual selector witness) * (LeftAuthority2432.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52632

namespace LeftBound52633
def owner : Owner := ⟨.program ⟨214⟩, ⟨12580⟩⟩
def transferEvent : Nat := 52633
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9930⟩⟩], []⟩ [⟨.result 2433 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2433 .coefficient)
      LeftAuthority2432.bound (LeftAuthority2432.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9930⟩⟩) (rawTerms := some (Proof.Events009.exact2433RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2432.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2432.bound []
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2432.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound52633

namespace LeftBound52634
def owner : Owner := ⟨.program ⟨214⟩, ⟨12580⟩⟩
def transferEvent : Nat := 52634
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 52629 .summary) (.transfer 52633) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52629 .summary)
      LeftBound52627.bound (LeftBound52627.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12579⟩⟩) (rawTerms := some (Proof.Events205.exact52629RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52627.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 52633)
      LeftBound52633.bound (LeftBound52633.actual selector witness) := by
  exact .transfer (LeftBound52633.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound52627.bound LeftBound52633.bound
def bound : CoeffClass := .finite ⟨34944, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52627.bound, LeftBound52633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound52627.actual selector witness) * (LeftBound52633.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52634

namespace LeftBound52640
def owner : Owner := ⟨.program ⟨214⟩, ⟨9931⟩⟩
def transferEvent : Nat := 52640
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 52638 .coefficient) (.predecessor 1 52639 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52638 .coefficient)
      LeftAuthority2432.bound (LeftAuthority2432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2432.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52639 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2432.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2432.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2432.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound52640

namespace LeftBound52645
def owner : Owner := ⟨.program ⟨214⟩, ⟨7260⟩⟩
def transferEvent : Nat := 52645
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 52643 .coefficient) (.predecessor 1 52644 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52643 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52644 .coefficient)
      LeftBound8516.bound (LeftBound8516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events033.exact8517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8516.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound8516.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound8516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound8516.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound52645

namespace LeftBound52650
def owner : Owner := ⟨.program ⟨214⟩, ⟨9932⟩⟩
def transferEvent : Nat := 52650
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 52648 .coefficient, .predecessor 1 52649 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 52648 .coefficient)
      LeftBound52645.bound (LeftBound52645.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52645.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52645.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 52649 .coefficient)
      LeftBound52640.bound (LeftBound52640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events205.exact52642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound52640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound52640.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound52645.bound, LeftBound52640.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52645.bound, LeftBound52640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound52645.actual selector witness, LeftBound52640.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound52650

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
