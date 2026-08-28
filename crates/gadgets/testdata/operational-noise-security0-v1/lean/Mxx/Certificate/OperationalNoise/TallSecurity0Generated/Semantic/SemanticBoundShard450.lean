import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard449

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66540
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def transferEvent : Nat := 66540
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66538 .coefficient) (.predecessor 1 66539 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66538 .coefficient)
      LeftBound66533.bound (LeftBound66533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66539 .coefficient)
      LeftAuthority66259.bound (LeftAuthority66259.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66259.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66533.bound LeftAuthority66259.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66533.bound, LeftAuthority66259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66533.actual selector witness) * (LeftAuthority66259.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66540

namespace LeftBound66541
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def transferEvent : Nat := 66541
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29589⟩⟩]⟩ [⟨.result 66260 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66260 .coefficient)
      LeftAuthority66259.bound (LeftAuthority66259.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29589⟩⟩) (rawTerms := some (Proof.Events258.exact66260RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66259.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66259.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66259.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66259.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66259.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66541

namespace LeftBound66542
def owner : Owner := ⟨.program ⟨214⟩, ⟨29591⟩⟩
def transferEvent : Nat := 66542
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66537 .summary) (.transfer 66541) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66537 .summary)
      LeftBound66536.bound (LeftBound66536.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25601⟩⟩) (rawTerms := some (Proof.Events259.exact66537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66541)
      LeftBound66541.bound (LeftBound66541.actual selector witness) := by
  exact .transfer (LeftBound66541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66536.bound LeftBound66541.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66536.bound, LeftBound66541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66536.actual selector witness) * (LeftBound66541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66542

namespace LeftBound66553
def owner : Owner := ⟨.program ⟨214⟩, ⟨22550⟩⟩
def transferEvent : Nat := 66553
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 66551 .coefficient) (.value (.predecessor 1 66552 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66551 .coefficient)
      LeftAuthority66549.bound (LeftAuthority66549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66549.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66552 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66549.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66549.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66549.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66553

namespace LeftBound66557
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def transferEvent : Nat := 66557
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66555 .coefficient) (.predecessor 1 66556 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66555 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66556 .coefficient)
      LeftBound66553.bound (LeftBound66553.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66554RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66553.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66553.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound66553.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound66553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound66553.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66557

namespace LeftBound66558
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def transferEvent : Nat := 66558
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22548⟩⟩]⟩ [⟨.result 66550 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66550 .coefficient)
      LeftAuthority66549.bound (LeftAuthority66549.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22548⟩⟩) (rawTerms := some (Proof.Events259.exact66550RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66549.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66549.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66549.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66558

namespace LeftBound66559
def owner : Owner := ⟨.program ⟨214⟩, ⟨22551⟩⟩
def transferEvent : Nat := 66559
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 66558) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66558)
      LeftBound66558.bound (LeftBound66558.actual selector witness) := by
  exact .transfer (LeftBound66558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound66558.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound66558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound66558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66559

namespace LeftBound66654
def owner : Owner := ⟨.program ⟨214⟩, ⟨16749⟩⟩
def transferEvent : Nat := 66654
def frameStart : Nat := 66615
def rule : BoundRule := .identity (.predecessor 0 66653 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66653 .coefficient)
      LeftAuthority66651.bound (LeftAuthority66651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66651.derived selector witness)

def rawBound : CoeffClass := LeftAuthority66651.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority66651.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66654

namespace LeftBound66671
def owner : Owner := ⟨.program ⟨214⟩, ⟨16823⟩⟩
def transferEvent : Nat := 66671
def frameStart : Nat := 66615
def rule : BoundRule := .sum [.predecessor 0 66669 .coefficient, .predecessor 1 66670 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66669 .coefficient)
      LeftBound66654.bound (LeftBound66654.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66670 .coefficient)
      LeftAuthority66667.bound (LeftAuthority66667.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66667.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66654.bound, LeftAuthority66667.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66654.bound, LeftAuthority66667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66654.actual selector witness, LeftAuthority66667.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66671

namespace LeftBound66674
def owner : Owner := ⟨.program ⟨214⟩, ⟨16824⟩⟩
def transferEvent : Nat := 66674
def frameStart : Nat := 66615
def rule : BoundRule := .identity (.predecessor 0 66673 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66673 .coefficient)
      LeftBound66671.bound (LeftBound66671.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66671.derived selector witness)

def rawBound : CoeffClass := LeftBound66671.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66671.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66674

namespace LeftBound66680
def owner : Owner := ⟨.program ⟨214⟩, ⟨16825⟩⟩
def transferEvent : Nat := 66680
def frameStart : Nat := 66615
def rule : BoundRule := .product (.predecessor 0 66678 .coefficient) (.predecessor 1 66679 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66678 .coefficient)
      LeftAuthority66676.bound (LeftAuthority66676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66679 .coefficient)
      LeftBound66674.bound (LeftBound66674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66674.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority66676.bound LeftBound66674.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66676.bound, LeftBound66674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority66676.actual selector witness) * (LeftBound66674.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66680

namespace LeftBound66688
def owner : Owner := ⟨.program ⟨214⟩, ⟨16826⟩⟩
def transferEvent : Nat := 66688
def frameStart : Nat := 66615
def rule : BoundRule := .sum [.predecessor 0 66686 .coefficient, .predecessor 1 66687 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66686 .coefficient)
      LeftAuthority66684.bound (LeftAuthority66684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66687 .coefficient)
      LeftBound66680.bound (LeftBound66680.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66682RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66680.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66680.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66684.bound, LeftBound66680.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66684.bound, LeftBound66680.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66684.actual selector witness, LeftBound66680.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66688

namespace LeftBound66692
def owner : Owner := ⟨.program ⟨214⟩, ⟨29590⟩⟩
def transferEvent : Nat := 66692
def frameStart : Nat := 66615
def rule : BoundRule := .product (.predecessor 0 66690 .coefficient) (.predecessor 1 66691 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66690 .coefficient)
      LeftBound66688.bound (LeftBound66688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66691 .coefficient)
      LeftAuthority66665.bound (LeftAuthority66665.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66665.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66665.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66688.bound LeftAuthority66665.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66688.bound, LeftAuthority66665.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66688.actual selector witness) * (LeftAuthority66665.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66692

namespace LeftBound66703
def owner : Owner := ⟨.program ⟨214⟩, ⟨16796⟩⟩
def transferEvent : Nat := 66703
def frameStart : Nat := 66615
def rule : BoundRule := .product (.predecessor 0 66701 .coefficient) (.predecessor 1 66702 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66701 .coefficient)
      LeftAuthority66676.bound (LeftAuthority66676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66702 .coefficient)
      LeftAuthority66699.bound (LeftAuthority66699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66699.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66699.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66676.bound LeftAuthority66699.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66676.bound, LeftAuthority66699.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66676.actual selector witness) * (LeftAuthority66699.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66703

namespace LeftBound66711
def owner : Owner := ⟨.program ⟨214⟩, ⟨16797⟩⟩
def transferEvent : Nat := 66711
def frameStart : Nat := 66615
def rule : BoundRule := .sum [.predecessor 0 66709 .coefficient, .predecessor 1 66710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66709 .coefficient)
      LeftAuthority66707.bound (LeftAuthority66707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66707.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66710 .coefficient)
      LeftBound66703.bound (LeftBound66703.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66703.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66703.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66707.bound, LeftBound66703.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66707.bound, LeftBound66703.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66707.actual selector witness, LeftBound66703.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66711

namespace LeftBound66715
def owner : Owner := ⟨.program ⟨214⟩, ⟨29594⟩⟩
def transferEvent : Nat := 66715
def frameStart : Nat := 66615
def rule : BoundRule := .sum [.predecessor 0 66713 .coefficient, .predecessor 1 66714 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66713 .coefficient)
      LeftBound66711.bound (LeftBound66711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66712RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66714 .coefficient)
      LeftBound66692.bound (LeftBound66692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66711.bound, LeftBound66692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66711.bound, LeftBound66692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66711.actual selector witness, LeftBound66692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66715

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
