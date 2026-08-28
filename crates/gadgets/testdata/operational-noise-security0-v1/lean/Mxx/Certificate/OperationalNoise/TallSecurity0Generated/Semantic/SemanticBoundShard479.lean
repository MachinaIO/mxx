import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard478

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70396
def owner : Owner := ⟨.program ⟨214⟩, ⟨27855⟩⟩
def transferEvent : Nat := 70396
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70394 .coefficient) (.predecessor 1 70395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70394 .coefficient)
      LeftBound70389.bound (LeftBound70389.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events274.exact70393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70389.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70389.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70395 .coefficient)
      LeftAuthority70115.bound (LeftAuthority70115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events273.exact70116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70389.bound LeftAuthority70115.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70389.bound, LeftAuthority70115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70389.actual selector witness) * (LeftAuthority70115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70396

namespace LeftBound70397
def owner : Owner := ⟨.program ⟨214⟩, ⟨27855⟩⟩
def transferEvent : Nat := 70397
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27853⟩⟩]⟩ [⟨.result 70116 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70116 .coefficient)
      LeftAuthority70115.bound (LeftAuthority70115.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27853⟩⟩) (rawTerms := some (Proof.Events273.exact70116RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70115.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70115.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70115.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70397

namespace LeftBound70398
def owner : Owner := ⟨.program ⟨214⟩, ⟨27855⟩⟩
def transferEvent : Nat := 70398
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70393 .summary) (.transfer 70397) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70393 .summary)
      LeftBound70392.bound (LeftBound70392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26063⟩⟩) (rawTerms := some (Proof.Events274.exact70393RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70397)
      LeftBound70397.bound (LeftBound70397.actual selector witness) := by
  exact .transfer (LeftBound70397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70392.bound LeftBound70397.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70392.bound, LeftBound70397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70392.actual selector witness) * (LeftBound70397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70398

namespace LeftBound70409
def owner : Owner := ⟨.program ⟨214⟩, ⟨21398⟩⟩
def transferEvent : Nat := 70409
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 70407 .coefficient) (.value (.predecessor 1 70408 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70407 .coefficient)
      LeftAuthority70405.bound (LeftAuthority70405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70408 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70405.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70405.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70405.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70409

namespace LeftBound70413
def owner : Owner := ⟨.program ⟨214⟩, ⟨21399⟩⟩
def transferEvent : Nat := 70413
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70411 .coefficient) (.predecessor 1 70412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70411 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70412 .coefficient)
      LeftBound70409.bound (LeftBound70409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70409.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound70409.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound70409.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound70409.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70413

namespace LeftBound70414
def owner : Owner := ⟨.program ⟨214⟩, ⟨21399⟩⟩
def transferEvent : Nat := 70414
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21396⟩⟩]⟩ [⟨.result 70406 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70406 .coefficient)
      LeftAuthority70405.bound (LeftAuthority70405.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21396⟩⟩) (rawTerms := some (Proof.Events275.exact70406RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70405.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70405.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70405.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70414

namespace LeftBound70415
def owner : Owner := ⟨.program ⟨214⟩, ⟨21399⟩⟩
def transferEvent : Nat := 70415
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 70414) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70414)
      LeftBound70414.bound (LeftBound70414.actual selector witness) := by
  exact .transfer (LeftBound70414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound70414.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound70414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound70414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70415

namespace LeftBound70510
def owner : Owner := ⟨.program ⟨214⟩, ⟨15937⟩⟩
def transferEvent : Nat := 70510
def frameStart : Nat := 70471
def rule : BoundRule := .identity (.predecessor 0 70509 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70509 .coefficient)
      LeftAuthority70507.bound (LeftAuthority70507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70507.derived selector witness)

def rawBound : CoeffClass := LeftAuthority70507.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority70507.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70510

namespace LeftBound70527
def owner : Owner := ⟨.program ⟨214⟩, ⟨16011⟩⟩
def transferEvent : Nat := 70527
def frameStart : Nat := 70471
def rule : BoundRule := .sum [.predecessor 0 70525 .coefficient, .predecessor 1 70526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70525 .coefficient)
      LeftBound70510.bound (LeftBound70510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70526 .coefficient)
      LeftAuthority70523.bound (LeftAuthority70523.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70510.bound, LeftAuthority70523.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70510.bound, LeftAuthority70523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70510.actual selector witness, LeftAuthority70523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70527

namespace LeftBound70530
def owner : Owner := ⟨.program ⟨214⟩, ⟨16012⟩⟩
def transferEvent : Nat := 70530
def frameStart : Nat := 70471
def rule : BoundRule := .identity (.predecessor 0 70529 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70529 .coefficient)
      LeftBound70527.bound (LeftBound70527.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70527.derived selector witness)

def rawBound : CoeffClass := LeftBound70527.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70527.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70530

namespace LeftBound70536
def owner : Owner := ⟨.program ⟨214⟩, ⟨16013⟩⟩
def transferEvent : Nat := 70536
def frameStart : Nat := 70471
def rule : BoundRule := .product (.predecessor 0 70534 .coefficient) (.predecessor 1 70535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70534 .coefficient)
      LeftAuthority70532.bound (LeftAuthority70532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70535 .coefficient)
      LeftBound70530.bound (LeftBound70530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70530.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority70532.bound LeftBound70530.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70532.bound, LeftBound70530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority70532.actual selector witness) * (LeftBound70530.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70536

namespace LeftBound70544
def owner : Owner := ⟨.program ⟨214⟩, ⟨16014⟩⟩
def transferEvent : Nat := 70544
def frameStart : Nat := 70471
def rule : BoundRule := .sum [.predecessor 0 70542 .coefficient, .predecessor 1 70543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70542 .coefficient)
      LeftAuthority70540.bound (LeftAuthority70540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70543 .coefficient)
      LeftBound70536.bound (LeftBound70536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70540.bound, LeftBound70536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70540.bound, LeftBound70536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70540.actual selector witness, LeftBound70536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70544

namespace LeftBound70548
def owner : Owner := ⟨.program ⟨214⟩, ⟨27854⟩⟩
def transferEvent : Nat := 70548
def frameStart : Nat := 70471
def rule : BoundRule := .product (.predecessor 0 70546 .coefficient) (.predecessor 1 70547 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70546 .coefficient)
      LeftBound70544.bound (LeftBound70544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70547 .coefficient)
      LeftAuthority70521.bound (LeftAuthority70521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70544.bound LeftAuthority70521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70544.bound, LeftAuthority70521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70544.actual selector witness) * (LeftAuthority70521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70548

namespace LeftBound70559
def owner : Owner := ⟨.program ⟨214⟩, ⟨15984⟩⟩
def transferEvent : Nat := 70559
def frameStart : Nat := 70471
def rule : BoundRule := .product (.predecessor 0 70557 .coefficient) (.predecessor 1 70558 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70557 .coefficient)
      LeftAuthority70532.bound (LeftAuthority70532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70558 .coefficient)
      LeftAuthority70555.bound (LeftAuthority70555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70555.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70555.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70532.bound LeftAuthority70555.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70532.bound, LeftAuthority70555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70532.actual selector witness) * (LeftAuthority70555.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70559

namespace LeftBound70567
def owner : Owner := ⟨.program ⟨214⟩, ⟨15985⟩⟩
def transferEvent : Nat := 70567
def frameStart : Nat := 70471
def rule : BoundRule := .sum [.predecessor 0 70565 .coefficient, .predecessor 1 70566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70565 .coefficient)
      LeftAuthority70563.bound (LeftAuthority70563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70566 .coefficient)
      LeftBound70559.bound (LeftBound70559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority70563.bound, LeftBound70559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70563.bound, LeftBound70559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority70563.actual selector witness, LeftBound70559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70567

namespace LeftBound70571
def owner : Owner := ⟨.program ⟨214⟩, ⟨27858⟩⟩
def transferEvent : Nat := 70571
def frameStart : Nat := 70471
def rule : BoundRule := .sum [.predecessor 0 70569 .coefficient, .predecessor 1 70570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70569 .coefficient)
      LeftBound70567.bound (LeftBound70567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70570 .coefficient)
      LeftBound70548.bound (LeftBound70548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70567.bound, LeftBound70548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70567.bound, LeftBound70548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70567.actual selector witness, LeftBound70548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70571

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
