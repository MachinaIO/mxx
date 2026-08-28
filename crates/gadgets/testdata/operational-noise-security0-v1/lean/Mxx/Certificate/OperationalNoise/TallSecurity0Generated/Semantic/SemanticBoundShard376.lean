import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard375

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound55554
def owner : Owner := ⟨.program ⟨214⟩, ⟨14223⟩⟩
def transferEvent : Nat := 55554
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55549 .summary) (.transfer 55553) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55549 .summary)
      LeftBound55547.bound (LeftBound55547.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14222⟩⟩) (rawTerms := some (Proof.Events216.exact55549RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55553)
      LeftBound55553.bound (LeftBound55553.actual selector witness) := by
  exact .transfer (LeftBound55553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55547.bound LeftBound55553.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55547.bound, LeftBound55553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55547.actual selector witness) * (LeftBound55553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55554

namespace LeftBound55562
def owner : Owner := ⟨.program ⟨214⟩, ⟨14224⟩⟩
def transferEvent : Nat := 55562
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 55560 .coefficient, .predecessor 1 55561 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55560 .coefficient)
      LeftBound55552.bound (LeftBound55552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55552.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55561 .coefficient)
      LeftBound55524.bound (LeftBound55524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55552.bound, LeftBound55524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55552.bound, LeftBound55524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55552.actual selector witness, LeftBound55524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55562

namespace LeftBound55564
def owner : Owner := ⟨.program ⟨214⟩, ⟨14224⟩⟩
def transferEvent : Nat := 55564
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 55559 .summary, .result 55529 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55559 .summary)
      LeftBound55554.bound (LeftBound55554.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14223⟩⟩) (rawTerms := some (Proof.Events217.exact55559RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55554.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55529 .summary)
      LeftBound55526.bound (LeftBound55526.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14219⟩⟩) (rawTerms := some (Proof.Events216.exact55529RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55554.bound, LeftBound55526.bound]
def bound : CoeffClass := .finite ⟨95435392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55554.bound, LeftBound55526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55554.actual selector witness, LeftBound55526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55564

namespace LeftBound55568
def owner : Owner := ⟨.program ⟨214⟩, ⟨26072⟩⟩
def transferEvent : Nat := 55568
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55566 .coefficient) (.predecessor 1 55567 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55566 .coefficient)
      LeftBound55562.bound (LeftBound55562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55565RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55562.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55567 .coefficient)
      LeftAuthority55500.bound (LeftAuthority55500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events216.exact55501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55500.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55500.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55562.bound LeftAuthority55500.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55562.bound, LeftAuthority55500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55562.actual selector witness) * (LeftAuthority55500.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55568

namespace LeftBound55569
def owner : Owner := ⟨.program ⟨214⟩, ⟨26072⟩⟩
def transferEvent : Nat := 55569
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26071⟩⟩]⟩ [⟨.result 55501 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55501 .coefficient)
      LeftAuthority55500.bound (LeftAuthority55500.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26071⟩⟩) (rawTerms := some (Proof.Events216.exact55501RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55500.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55500.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55500.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55500.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55500.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55569

namespace LeftBound55570
def owner : Owner := ⟨.program ⟨214⟩, ⟨26072⟩⟩
def transferEvent : Nat := 55570
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55565 .summary) (.transfer 55569) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55565 .summary)
      LeftBound55564.bound (LeftBound55564.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14224⟩⟩) (rawTerms := some (Proof.Events217.exact55565RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55564.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55569)
      LeftBound55569.bound (LeftBound55569.actual selector witness) := by
  exact .transfer (LeftBound55569.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55564.bound LeftBound55569.bound
def bound : CoeffClass := .finite ⟨350249415606272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55564.bound, LeftBound55569.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55564.actual selector witness) * (LeftBound55569.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55570

namespace LeftBound55581
def owner : Owner := ⟨.program ⟨214⟩, ⟨19534⟩⟩
def transferEvent : Nat := 55581
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 55579 .coefficient) (.value (.predecessor 1 55580 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55579 .coefficient)
      LeftAuthority55577.bound (LeftAuthority55577.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55577.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55580 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55577.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55577.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55577.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55581

namespace LeftBound55585
def owner : Owner := ⟨.program ⟨214⟩, ⟨19535⟩⟩
def transferEvent : Nat := 55585
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 55583 .coefficient) (.predecessor 1 55584 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55583 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55584 .coefficient)
      LeftBound55581.bound (LeftBound55581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55582RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55581.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound55581.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound55581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound55581.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55585

namespace LeftBound55586
def owner : Owner := ⟨.program ⟨214⟩, ⟨19535⟩⟩
def transferEvent : Nat := 55586
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19532⟩⟩]⟩ [⟨.result 55578 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55578 .coefficient)
      LeftAuthority55577.bound (LeftAuthority55577.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19532⟩⟩) (rawTerms := some (Proof.Events217.exact55578RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55577.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority55577.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55577.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound55586

namespace LeftBound55587
def owner : Owner := ⟨.program ⟨214⟩, ⟨19535⟩⟩
def transferEvent : Nat := 55587
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 55586) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 55586)
      LeftBound55586.bound (LeftBound55586.actual selector witness) := by
  exact .transfer (LeftBound55586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound55586.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound55586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound55586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55587

namespace LeftBound55666
def owner : Owner := ⟨.program ⟨214⟩, ⟨14217⟩⟩
def transferEvent : Nat := 55666
def frameStart : Nat := 55637
def rule : BoundRule := .product (.predecessor 0 55664 .coefficient) (.predecessor 1 55665 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55664 .coefficient)
      LeftAuthority55662.bound (LeftAuthority55662.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55665 .coefficient)
      LeftAuthority55659.bound (LeftAuthority55659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55659.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority55662.bound LeftAuthority55659.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55662.bound, LeftAuthority55659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority55662.actual selector witness) * (LeftAuthority55659.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55666

namespace LeftBound55670
def owner : Owner := ⟨.program ⟨214⟩, ⟨14218⟩⟩
def transferEvent : Nat := 55670
def frameStart : Nat := 55637
def rule : BoundRule := .identity (.predecessor 0 55669 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55669 .coefficient)
      LeftBound55666.bound (LeftBound55666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55666.derived selector witness)

def rawBound : CoeffClass := LeftBound55666.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55666.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55670

namespace LeftBound55687
def owner : Owner := ⟨.program ⟨214⟩, ⟨14318⟩⟩
def transferEvent : Nat := 55687
def frameStart : Nat := 55637
def rule : BoundRule := .sum [.predecessor 0 55685 .coefficient, .predecessor 1 55686 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55685 .coefficient)
      LeftBound55670.bound (LeftBound55670.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55686 .coefficient)
      LeftAuthority55683.bound (LeftAuthority55683.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55683.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound55670.bound, LeftAuthority55683.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55670.bound, LeftAuthority55683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound55670.actual selector witness, LeftAuthority55683.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound55687

namespace LeftBound55690
def owner : Owner := ⟨.program ⟨214⟩, ⟨14319⟩⟩
def transferEvent : Nat := 55690
def frameStart : Nat := 55637
def rule : BoundRule := .identity (.predecessor 0 55689 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55689 .coefficient)
      LeftBound55687.bound (LeftBound55687.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound55687.derived selector witness)

def rawBound : CoeffClass := LeftBound55687.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound55687.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound55690

namespace LeftBound55696
def owner : Owner := ⟨.program ⟨214⟩, ⟨14320⟩⟩
def transferEvent : Nat := 55696
def frameStart : Nat := 55637
def rule : BoundRule := .product (.predecessor 0 55694 .coefficient) (.predecessor 1 55695 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55694 .coefficient)
      LeftAuthority55692.bound (LeftAuthority55692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55695 .coefficient)
      LeftBound55690.bound (LeftBound55690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55690.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority55692.bound LeftBound55690.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55692.bound, LeftBound55690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority55692.actual selector witness) * (LeftBound55690.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound55696

namespace LeftBound55712
def owner : Owner := ⟨.program ⟨214⟩, ⟨7853⟩⟩
def transferEvent : Nat := 55712
def frameStart : Nat := 55637
def rule : BoundRule := .scale (.predecessor 0 55710 .coefficient) (.value (.predecessor 1 55711 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 55710 .coefficient)
      LeftAuthority55708.bound (LeftAuthority55708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events217.exact55709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority55708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority55708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 55711 .coefficient)
      LeftAuthority55699.bound (LeftAuthority55699.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority55699.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority55708.bound LeftAuthority55699.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority55708.bound, LeftAuthority55699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority55708.actual selector witness) * (LeftAuthority55699.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound55712

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
