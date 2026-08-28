import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard418

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62474
def owner : Owner := ⟨.program ⟨214⟩, ⟨21766⟩⟩
def transferEvent : Nat := 62474
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 62472 .coefficient) (.value (.predecessor 1 62473 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62472 .coefficient)
      LeftAuthority62470.bound (LeftAuthority62470.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62471RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62470.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62473 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62470.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62470.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62470.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62474

namespace LeftBound62478
def owner : Owner := ⟨.program ⟨214⟩, ⟨21767⟩⟩
def transferEvent : Nat := 62478
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62476 .coefficient) (.predecessor 1 62477 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62476 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62477 .coefficient)
      LeftBound62474.bound (LeftBound62474.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62475RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62474.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62474.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound62474.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound62474.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound62474.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62478

namespace LeftBound62479
def owner : Owner := ⟨.program ⟨214⟩, ⟨21767⟩⟩
def transferEvent : Nat := 62479
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21764⟩⟩]⟩ [⟨.result 62471 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62471 .coefficient)
      LeftAuthority62470.bound (LeftAuthority62470.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21764⟩⟩) (rawTerms := some (Proof.Events244.exact62471RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62470.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62470.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62470.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62470.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62470.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62479

namespace LeftBound62480
def owner : Owner := ⟨.program ⟨214⟩, ⟨21767⟩⟩
def transferEvent : Nat := 62480
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 62479) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62479)
      LeftBound62479.bound (LeftBound62479.actual selector witness) := by
  exact .transfer (LeftBound62479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound62479.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound62479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound62479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62480

namespace LeftBound62575
def owner : Owner := ⟨.program ⟨214⟩, ⟨16267⟩⟩
def transferEvent : Nat := 62575
def frameStart : Nat := 62536
def rule : BoundRule := .identity (.predecessor 0 62574 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62574 .coefficient)
      LeftAuthority62572.bound (LeftAuthority62572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62572.derived selector witness)

def rawBound : CoeffClass := LeftAuthority62572.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority62572.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62575

namespace LeftBound62592
def owner : Owner := ⟨.program ⟨214⟩, ⟨16341⟩⟩
def transferEvent : Nat := 62592
def frameStart : Nat := 62536
def rule : BoundRule := .sum [.predecessor 0 62590 .coefficient, .predecessor 1 62591 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62590 .coefficient)
      LeftBound62575.bound (LeftBound62575.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62591 .coefficient)
      LeftAuthority62588.bound (LeftAuthority62588.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority62588.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62575.bound, LeftAuthority62588.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62575.bound, LeftAuthority62588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62575.actual selector witness, LeftAuthority62588.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62592

namespace LeftBound62595
def owner : Owner := ⟨.program ⟨214⟩, ⟨16342⟩⟩
def transferEvent : Nat := 62595
def frameStart : Nat := 62536
def rule : BoundRule := .identity (.predecessor 0 62594 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62594 .coefficient)
      LeftBound62592.bound (LeftBound62592.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62592.derived selector witness)

def rawBound : CoeffClass := LeftBound62592.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62592.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound62592.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62595

namespace LeftBound62601
def owner : Owner := ⟨.program ⟨214⟩, ⟨16343⟩⟩
def transferEvent : Nat := 62601
def frameStart : Nat := 62536
def rule : BoundRule := .product (.predecessor 0 62599 .coefficient) (.predecessor 1 62600 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62599 .coefficient)
      LeftAuthority62597.bound (LeftAuthority62597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62600 .coefficient)
      LeftBound62595.bound (LeftBound62595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62595.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62595.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority62597.bound LeftBound62595.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62597.bound, LeftBound62595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority62597.actual selector witness) * (LeftBound62595.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62601

namespace LeftBound62609
def owner : Owner := ⟨.program ⟨214⟩, ⟨16344⟩⟩
def transferEvent : Nat := 62609
def frameStart : Nat := 62536
def rule : BoundRule := .sum [.predecessor 0 62607 .coefficient, .predecessor 1 62608 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62607 .coefficient)
      LeftAuthority62605.bound (LeftAuthority62605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62605.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62605.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62608 .coefficient)
      LeftBound62601.bound (LeftBound62601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62603RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62605.bound, LeftBound62601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62605.bound, LeftBound62601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62605.actual selector witness, LeftBound62601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62609

namespace LeftBound62613
def owner : Owner := ⟨.program ⟨214⟩, ⟨28524⟩⟩
def transferEvent : Nat := 62613
def frameStart : Nat := 62536
def rule : BoundRule := .product (.predecessor 0 62611 .coefficient) (.predecessor 1 62612 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62611 .coefficient)
      LeftBound62609.bound (LeftBound62609.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62609.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62609.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62612 .coefficient)
      LeftAuthority62586.bound (LeftAuthority62586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62586.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62586.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62609.bound LeftAuthority62586.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62609.bound, LeftAuthority62586.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62609.actual selector witness) * (LeftAuthority62586.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62613

namespace LeftBound62624
def owner : Owner := ⟨.program ⟨214⟩, ⟨17612⟩⟩
def transferEvent : Nat := 62624
def frameStart : Nat := 62536
def rule : BoundRule := .product (.predecessor 0 62622 .coefficient) (.predecessor 1 62623 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62622 .coefficient)
      LeftAuthority62597.bound (LeftAuthority62597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62623 .coefficient)
      LeftAuthority62620.bound (LeftAuthority62620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62620.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority62597.bound LeftAuthority62620.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62597.bound, LeftAuthority62620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority62597.actual selector witness) * (LeftAuthority62620.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62624

namespace LeftBound62632
def owner : Owner := ⟨.program ⟨214⟩, ⟨17613⟩⟩
def transferEvent : Nat := 62632
def frameStart : Nat := 62536
def rule : BoundRule := .sum [.predecessor 0 62630 .coefficient, .predecessor 1 62631 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62630 .coefficient)
      LeftAuthority62628.bound (LeftAuthority62628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62628.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62631 .coefficient)
      LeftBound62624.bound (LeftBound62624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62626RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62624.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62628.bound, LeftBound62624.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62628.bound, LeftBound62624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62628.actual selector witness, LeftBound62624.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62632

namespace LeftBound62636
def owner : Owner := ⟨.program ⟨214⟩, ⟨28529⟩⟩
def transferEvent : Nat := 62636
def frameStart : Nat := 62536
def rule : BoundRule := .sum [.predecessor 0 62634 .coefficient, .predecessor 1 62635 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62634 .coefficient)
      LeftBound62632.bound (LeftBound62632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62633RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62635 .coefficient)
      LeftBound62613.bound (LeftBound62613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62632.bound, LeftBound62613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62632.bound, LeftBound62613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62632.actual selector witness, LeftBound62613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62636

namespace LeftBound62649
def owner : Owner := ⟨.program ⟨214⟩, ⟨28526⟩⟩
def transferEvent : Nat := 62649
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62647 .coefficient, .predecessor 1 62648 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62647 .coefficient)
      LeftBound62478.bound (LeftBound62478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62646RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62478.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62478.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62648 .coefficient)
      LeftBound62461.bound (LeftBound62461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62461.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62461.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62478.bound, LeftBound62461.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62478.bound, LeftBound62461.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62478.actual selector witness, LeftBound62461.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62649

namespace LeftBound62652
def owner : Owner := ⟨.program ⟨214⟩, ⟨28526⟩⟩
def transferEvent : Nat := 62652
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62646 .summary, .result 62468 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62646 .summary)
      LeftBound62480.bound (LeftBound62480.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21767⟩⟩) (rawTerms := some (Proof.Events244.exact62646RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62480.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62468 .summary)
      LeftBound62463.bound (LeftBound62463.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28525⟩⟩) (rawTerms := some (Proof.Events244.exact62468RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62463.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62480.bound, LeftBound62463.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62480.bound, LeftBound62463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62480.actual selector witness, LeftBound62463.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62652

namespace LeftBound62656
def owner : Owner := ⟨.program ⟨214⟩, ⟨28527⟩⟩
def transferEvent : Nat := 62656
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62654 .coefficient) (.predecessor 1 62655 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62654 .coefficient)
      LeftBound62649.bound (LeftBound62649.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62649.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62649.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62655 .coefficient)
      LeftBound5658.bound (LeftBound5658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62649.bound LeftBound5658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62649.bound, LeftBound5658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62649.actual selector witness) * (LeftBound5658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62656

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
