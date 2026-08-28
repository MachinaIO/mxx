import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard195

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29366
def owner : Owner := ⟨.program ⟨214⟩, ⟨10787⟩⟩
def transferEvent : Nat := 29366
def frameStart : Nat := 29279
def rule : BoundRule := .sum [.predecessor 0 29364 .coefficient, .predecessor 1 29365 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29364 .coefficient)
      LeftBound29361.bound (LeftBound29361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29365 .coefficient)
      LeftBound29338.bound (LeftBound29338.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29340RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29338.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29361.bound, LeftBound29338.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29361.bound, LeftBound29338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29361.actual selector witness, LeftBound29338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29366

namespace LeftBound29370
def owner : Owner := ⟨.program ⟨214⟩, ⟨25006⟩⟩
def transferEvent : Nat := 29370
def frameStart : Nat := 29279
def rule : BoundRule := .product (.predecessor 0 29368 .coefficient) (.predecessor 1 29369 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29368 .coefficient)
      LeftBound29366.bound (LeftBound29366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29367RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29366.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29369 .coefficient)
      LeftAuthority29323.bound (LeftAuthority29323.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29324RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29323.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29323.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29366.bound LeftAuthority29323.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29366.bound, LeftAuthority29323.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29366.actual selector witness) * (LeftAuthority29323.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29370

namespace LeftBound29381
def owner : Owner := ⟨.program ⟨214⟩, ⟨14967⟩⟩
def transferEvent : Nat := 29381
def frameStart : Nat := 29279
def rule : BoundRule := .product (.predecessor 0 29379 .coefficient) (.predecessor 1 29380 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29379 .coefficient)
      LeftAuthority29334.bound (LeftAuthority29334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29380 .coefficient)
      LeftAuthority29377.bound (LeftAuthority29377.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29378RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29377.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29377.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority29334.bound LeftAuthority29377.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29334.bound, LeftAuthority29377.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority29334.actual selector witness) * (LeftAuthority29377.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29381

namespace LeftBound29389
def owner : Owner := ⟨.program ⟨214⟩, ⟨14968⟩⟩
def transferEvent : Nat := 29389
def frameStart : Nat := 29279
def rule : BoundRule := .sum [.predecessor 0 29387 .coefficient, .predecessor 1 29388 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29387 .coefficient)
      LeftAuthority29385.bound (LeftAuthority29385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29385.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29388 .coefficient)
      LeftBound29381.bound (LeftBound29381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29381.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29381.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority29385.bound, LeftBound29381.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29385.bound, LeftBound29381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority29385.actual selector witness, LeftBound29381.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29389

namespace LeftBound29393
def owner : Owner := ⟨.program ⟨214⟩, ⟨25007⟩⟩
def transferEvent : Nat := 29393
def frameStart : Nat := 29279
def rule : BoundRule := .sum [.predecessor 0 29391 .coefficient, .predecessor 1 29392 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29391 .coefficient)
      LeftBound29389.bound (LeftBound29389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29390RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29392 .coefficient)
      LeftBound29370.bound (LeftBound29370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29370.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29370.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29389.bound, LeftBound29370.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29389.bound, LeftBound29370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29389.actual selector witness, LeftBound29370.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29393

namespace LeftBound29406
def owner : Owner := ⟨.program ⟨214⟩, ⟨25005⟩⟩
def transferEvent : Nat := 29406
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29404 .coefficient, .predecessor 1 29405 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29404 .coefficient)
      LeftBound29227.bound (LeftBound29227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29405 .coefficient)
      LeftBound29210.bound (LeftBound29210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29210.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29210.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29227.bound, LeftBound29210.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29227.bound, LeftBound29210.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29227.actual selector witness, LeftBound29210.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29406

namespace LeftBound29409
def owner : Owner := ⟨.program ⟨214⟩, ⟨25005⟩⟩
def transferEvent : Nat := 29409
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29403 .summary, .result 29217 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29403 .summary)
      LeftBound29229.bound (LeftBound29229.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19111⟩⟩) (rawTerms := some (Proof.Events114.exact29403RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29229.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29217 .summary)
      LeftBound29212.bound (LeftBound29212.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25004⟩⟩) (rawTerms := some (Proof.Events114.exact29217RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29212.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29229.bound, LeftBound29212.bound]
def bound : CoeffClass := .finite ⟨352014917316608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29229.bound, LeftBound29212.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29229.actual selector witness, LeftBound29212.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29409

namespace LeftBound29413
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def transferEvent : Nat := 29413
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29411 .coefficient) (.predecessor 1 29412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29411 .coefficient)
      LeftBound29406.bound (LeftBound29406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29412 .coefficient)
      LeftAuthority29132.bound (LeftAuthority29132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events113.exact29133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29406.bound LeftAuthority29132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29406.bound, LeftAuthority29132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29406.actual selector witness) * (LeftAuthority29132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29413

namespace LeftBound29414
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def transferEvent : Nat := 29414
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26603⟩⟩]⟩ [⟨.result 29133 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29133 .coefficient)
      LeftAuthority29132.bound (LeftAuthority29132.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26603⟩⟩) (rawTerms := some (Proof.Events113.exact29133RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29132.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29132.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29132.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29414

namespace LeftBound29415
def owner : Owner := ⟨.program ⟨214⟩, ⟨26605⟩⟩
def transferEvent : Nat := 29415
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29410 .summary) (.transfer 29414) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29410 .summary)
      LeftBound29409.bound (LeftBound29409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25005⟩⟩) (rawTerms := some (Proof.Events114.exact29410RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29414)
      LeftBound29414.bound (LeftBound29414.actual selector witness) := by
  exact .transfer (LeftBound29414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29409.bound LeftBound29414.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29409.bound, LeftBound29414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29409.actual selector witness) * (LeftBound29414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29415

namespace LeftBound29426
def owner : Owner := ⟨.program ⟨214⟩, ⟨20550⟩⟩
def transferEvent : Nat := 29426
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 29424 .coefficient) (.value (.predecessor 1 29425 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29424 .coefficient)
      LeftAuthority29422.bound (LeftAuthority29422.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29423RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29422.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29425 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29422.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29422.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29422.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29426

namespace LeftBound29430
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def transferEvent : Nat := 29430
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29428 .coefficient) (.predecessor 1 29429 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29428 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29429 .coefficient)
      LeftBound29426.bound (LeftBound29426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events114.exact29427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29426.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29426.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound29426.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound29426.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound29426.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29430

namespace LeftBound29431
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def transferEvent : Nat := 29431
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20548⟩⟩]⟩ [⟨.result 29423 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29423 .coefficient)
      LeftAuthority29422.bound (LeftAuthority29422.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20548⟩⟩) (rawTerms := some (Proof.Events114.exact29423RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29422.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29422.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29422.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29422.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29422.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29431

namespace LeftBound29432
def owner : Owner := ⟨.program ⟨214⟩, ⟨20551⟩⟩
def transferEvent : Nat := 29432
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 29431) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29431)
      LeftBound29431.bound (LeftBound29431.actual selector witness) := by
  exact .transfer (LeftBound29431.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound29431.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound29431.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound29431.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29432

namespace LeftBound29527
def owner : Owner := ⟨.program ⟨214⟩, ⟨14966⟩⟩
def transferEvent : Nat := 29527
def frameStart : Nat := 29488
def rule : BoundRule := .identity (.predecessor 0 29526 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29526 .coefficient)
      LeftAuthority29524.bound (LeftAuthority29524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29524.derived selector witness)

def rawBound : CoeffClass := LeftAuthority29524.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority29524.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound29527

namespace LeftBound29544
def owner : Owner := ⟨.program ⟨214⟩, ⟨15005⟩⟩
def transferEvent : Nat := 29544
def frameStart : Nat := 29488
def rule : BoundRule := .sum [.predecessor 0 29542 .coefficient, .predecessor 1 29543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29542 .coefficient)
      LeftBound29527.bound (LeftBound29527.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound29527.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29543 .coefficient)
      LeftAuthority29540.bound (LeftAuthority29540.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority29540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29527.bound, LeftAuthority29540.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29527.bound, LeftAuthority29540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29527.actual selector witness, LeftAuthority29540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29544

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
