import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard673

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound98355
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def transferEvent : Nat := 98355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26130⟩⟩]⟩ [⟨.result 98287 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98287 .coefficient)
      LeftAuthority98286.bound (LeftAuthority98286.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26130⟩⟩) (rawTerms := some (Proof.Events383.exact98287RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98286.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98286.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98286.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98286.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98286.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98355

namespace LeftBound98356
def owner : Owner := ⟨.program ⟨214⟩, ⟨26131⟩⟩
def transferEvent : Nat := 98356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98351 .summary) (.transfer 98355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98351 .summary)
      LeftBound98350.bound (LeftBound98350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14405⟩⟩) (rawTerms := some (Proof.Events384.exact98351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98355)
      LeftBound98355.bound (LeftBound98355.actual selector witness) := by
  exact .transfer (LeftBound98355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98350.bound LeftBound98355.bound
def bound : CoeffClass := .finite ⟨350261629419520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98350.bound, LeftBound98355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98350.actual selector witness) * (LeftBound98355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98356

namespace LeftBound98367
def owner : Owner := ⟨.program ⟨214⟩, ⟨19591⟩⟩
def transferEvent : Nat := 98367
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 98365 .coefficient) (.value (.predecessor 1 98366 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98365 .coefficient)
      LeftAuthority98363.bound (LeftAuthority98363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98363.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98366 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98363.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98363.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98363.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98367

namespace LeftBound98371
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def transferEvent : Nat := 98371
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 98369 .coefficient) (.predecessor 1 98370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98369 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98370 .coefficient)
      LeftBound98367.bound (LeftBound98367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98367.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound98367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound98367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound98367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98371

namespace LeftBound98372
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def transferEvent : Nat := 98372
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19589⟩⟩]⟩ [⟨.result 98364 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 98364 .coefficient)
      LeftAuthority98363.bound (LeftAuthority98363.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19589⟩⟩) (rawTerms := some (Proof.Events384.exact98364RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98363.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority98363.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98363.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound98372

namespace LeftBound98373
def owner : Owner := ⟨.program ⟨214⟩, ⟨19592⟩⟩
def transferEvent : Nat := 98373
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 98372) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 98372)
      LeftBound98372.bound (LeftBound98372.actual selector witness) := by
  exact .transfer (LeftBound98372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound98372.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound98372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound98372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98373

namespace LeftBound98428
def owner : Owner := ⟨.program ⟨214⟩, ⟨14398⟩⟩
def transferEvent : Nat := 98428
def frameStart : Nat := 98411
def rule : BoundRule := .product (.predecessor 0 98426 .coefficient) (.predecessor 1 98427 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98426 .coefficient)
      LeftAuthority98424.bound (LeftAuthority98424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98424.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98424.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98427 .coefficient)
      LeftAuthority98421.bound (LeftAuthority98421.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98422RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98421.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98421.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority98424.bound LeftAuthority98421.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98424.bound, LeftAuthority98421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority98424.actual selector witness) * (LeftAuthority98421.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98428

namespace LeftBound98432
def owner : Owner := ⟨.program ⟨214⟩, ⟨14399⟩⟩
def transferEvent : Nat := 98432
def frameStart : Nat := 98411
def rule : BoundRule := .identity (.predecessor 0 98431 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98431 .coefficient)
      LeftBound98428.bound (LeftBound98428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98430RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98428.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98428.derived selector witness)

def rawBound : CoeffClass := LeftBound98428.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98428.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98432

namespace LeftBound98449
def owner : Owner := ⟨.program ⟨214⟩, ⟨14523⟩⟩
def transferEvent : Nat := 98449
def frameStart : Nat := 98411
def rule : BoundRule := .sum [.predecessor 0 98447 .coefficient, .predecessor 1 98448 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98447 .coefficient)
      LeftBound98432.bound (LeftBound98432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98448 .coefficient)
      LeftAuthority98445.bound (LeftAuthority98445.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98432.bound, LeftAuthority98445.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98432.bound, LeftAuthority98445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98432.actual selector witness, LeftAuthority98445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98449

namespace LeftBound98452
def owner : Owner := ⟨.program ⟨214⟩, ⟨14524⟩⟩
def transferEvent : Nat := 98452
def frameStart : Nat := 98411
def rule : BoundRule := .identity (.predecessor 0 98451 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98451 .coefficient)
      LeftBound98449.bound (LeftBound98449.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound98449.derived selector witness)

def rawBound : CoeffClass := LeftBound98449.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound98449.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98452

namespace LeftBound98458
def owner : Owner := ⟨.program ⟨214⟩, ⟨14525⟩⟩
def transferEvent : Nat := 98458
def frameStart : Nat := 98411
def rule : BoundRule := .product (.predecessor 0 98456 .coefficient) (.predecessor 1 98457 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98456 .coefficient)
      LeftAuthority98454.bound (LeftAuthority98454.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98455RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98454.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98454.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98457 .coefficient)
      LeftBound98452.bound (LeftBound98452.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98453RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98452.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98452.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority98454.bound LeftBound98452.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98454.bound, LeftBound98452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority98454.actual selector witness) * (LeftBound98452.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98458

namespace LeftBound98474
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 98474
def frameStart : Nat := 98411
def rule : BoundRule := .scale (.predecessor 0 98472 .coefficient) (.value (.predecessor 1 98473 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98472 .coefficient)
      LeftAuthority98470.bound (LeftAuthority98470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98473 .coefficient)
      LeftAuthority98461.bound (LeftAuthority98461.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority98461.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority98470.bound LeftAuthority98461.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98470.bound, LeftAuthority98461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority98470.actual selector witness) * (LeftAuthority98461.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound98474

namespace LeftBound98477
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 98477
def frameStart : Nat := 98411
def rule : BoundRule := .identity (.predecessor 0 98476 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98476 .coefficient)
      LeftAuthority98464.bound (LeftAuthority98464.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98465RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98464.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98464.derived selector witness)

def rawBound : CoeffClass := LeftAuthority98464.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority98464.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority98464.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound98477

namespace LeftBound98481
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 98481
def frameStart : Nat := 98411
def rule : BoundRule := .product (.predecessor 0 98479 .coefficient) (.predecessor 1 98480 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98479 .coefficient)
      LeftBound98477.bound (LeftBound98477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98477.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98480 .coefficient)
      LeftBound98474.bound (LeftBound98474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98474.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98477.bound LeftBound98474.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98477.bound, LeftBound98474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98477.actual selector witness) * (LeftBound98474.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98481

namespace LeftBound98486
def owner : Owner := ⟨.program ⟨214⟩, ⟨14526⟩⟩
def transferEvent : Nat := 98486
def frameStart : Nat := 98411
def rule : BoundRule := .sum [.predecessor 0 98484 .coefficient, .predecessor 1 98485 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98484 .coefficient)
      LeftBound98481.bound (LeftBound98481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98481.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98485 .coefficient)
      LeftBound98458.bound (LeftBound98458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98458.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98458.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound98481.bound, LeftBound98458.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98481.bound, LeftBound98458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound98481.actual selector witness, LeftBound98458.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound98486

namespace LeftBound98490
def owner : Owner := ⟨.program ⟨214⟩, ⟨26133⟩⟩
def transferEvent : Nat := 98490
def frameStart : Nat := 98411
def rule : BoundRule := .product (.predecessor 0 98488 .coefficient) (.predecessor 1 98489 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 98488 .coefficient)
      LeftBound98486.bound (LeftBound98486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98486.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 98489 .coefficient)
      LeftAuthority98443.bound (LeftAuthority98443.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98444RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority98443.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority98443.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound98486.bound LeftAuthority98443.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98486.bound, LeftAuthority98443.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound98486.actual selector witness) * (LeftAuthority98443.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound98490

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
