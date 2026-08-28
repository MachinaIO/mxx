import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard060
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard366

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54343
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def transferEvent : Nat := 54343
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21836⟩⟩]⟩ [⟨.result 54335 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54335 .coefficient)
      LeftAuthority54334.bound (LeftAuthority54334.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21836⟩⟩) (rawTerms := some (Proof.Events212.exact54335RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54334.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54334.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority54334.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54343

namespace LeftBound54344
def owner : Owner := ⟨.program ⟨214⟩, ⟨21839⟩⟩
def transferEvent : Nat := 54344
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 54343) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 54343)
      LeftBound54343.bound (LeftBound54343.actual selector witness) := by
  exact .transfer (LeftBound54343.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound54343.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound54343.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound54343.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54344

namespace LeftBound54439
def owner : Owner := ⟨.program ⟨214⟩, ⟨16267⟩⟩
def transferEvent : Nat := 54439
def frameStart : Nat := 54400
def rule : BoundRule := .identity (.predecessor 0 54438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54438 .coefficient)
      LeftAuthority54436.bound (LeftAuthority54436.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54437RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54436.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54436.derived selector witness)

def rawBound : CoeffClass := LeftAuthority54436.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority54436.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54439

namespace LeftBound54456
def owner : Owner := ⟨.program ⟨214⟩, ⟨16341⟩⟩
def transferEvent : Nat := 54456
def frameStart : Nat := 54400
def rule : BoundRule := .sum [.predecessor 0 54454 .coefficient, .predecessor 1 54455 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54454 .coefficient)
      LeftBound54439.bound (LeftBound54439.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54439.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54455 .coefficient)
      LeftAuthority54452.bound (LeftAuthority54452.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54452.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54439.bound, LeftAuthority54452.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54439.bound, LeftAuthority54452.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54439.actual selector witness, LeftAuthority54452.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54456

namespace LeftBound54459
def owner : Owner := ⟨.program ⟨214⟩, ⟨16342⟩⟩
def transferEvent : Nat := 54459
def frameStart : Nat := 54400
def rule : BoundRule := .identity (.predecessor 0 54458 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54458 .coefficient)
      LeftBound54456.bound (LeftBound54456.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54456.derived selector witness)

def rawBound : CoeffClass := LeftBound54456.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound54456.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54459

namespace LeftBound54465
def owner : Owner := ⟨.program ⟨214⟩, ⟨16343⟩⟩
def transferEvent : Nat := 54465
def frameStart : Nat := 54400
def rule : BoundRule := .product (.predecessor 0 54463 .coefficient) (.predecessor 1 54464 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54463 .coefficient)
      LeftAuthority54461.bound (LeftAuthority54461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54464 .coefficient)
      LeftBound54459.bound (LeftBound54459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54460RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54459.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority54461.bound LeftBound54459.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54461.bound, LeftBound54459.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority54461.actual selector witness) * (LeftBound54459.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54465

namespace LeftBound54473
def owner : Owner := ⟨.program ⟨214⟩, ⟨16344⟩⟩
def transferEvent : Nat := 54473
def frameStart : Nat := 54400
def rule : BoundRule := .sum [.predecessor 0 54471 .coefficient, .predecessor 1 54472 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54471 .coefficient)
      LeftAuthority54469.bound (LeftAuthority54469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54469.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54469.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54472 .coefficient)
      LeftBound54465.bound (LeftBound54465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54467RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54465.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54469.bound, LeftBound54465.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54469.bound, LeftBound54465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54469.actual selector witness, LeftBound54465.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54473

namespace LeftBound54477
def owner : Owner := ⟨.program ⟨214⟩, ⟨28531⟩⟩
def transferEvent : Nat := 54477
def frameStart : Nat := 54400
def rule : BoundRule := .product (.predecessor 0 54475 .coefficient) (.predecessor 1 54476 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54475 .coefficient)
      LeftBound54473.bound (LeftBound54473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54473.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54473.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54476 .coefficient)
      LeftAuthority54450.bound (LeftAuthority54450.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54451RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54450.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound54473.bound LeftAuthority54450.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54473.bound, LeftAuthority54450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound54473.actual selector witness) * (LeftAuthority54450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54477

namespace LeftBound54488
def owner : Owner := ⟨.program ⟨214⟩, ⟨16312⟩⟩
def transferEvent : Nat := 54488
def frameStart : Nat := 54400
def rule : BoundRule := .product (.predecessor 0 54486 .coefficient) (.predecessor 1 54487 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54486 .coefficient)
      LeftAuthority54461.bound (LeftAuthority54461.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54461.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54461.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54487 .coefficient)
      LeftAuthority54484.bound (LeftAuthority54484.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54485RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54484.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54484.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54461.bound LeftAuthority54484.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54461.bound, LeftAuthority54484.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority54461.actual selector witness) * (LeftAuthority54484.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54488

namespace LeftBound54496
def owner : Owner := ⟨.program ⟨214⟩, ⟨16313⟩⟩
def transferEvent : Nat := 54496
def frameStart : Nat := 54400
def rule : BoundRule := .sum [.predecessor 0 54494 .coefficient, .predecessor 1 54495 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54494 .coefficient)
      LeftAuthority54492.bound (LeftAuthority54492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54493RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54492.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54495 .coefficient)
      LeftBound54488.bound (LeftBound54488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54492.bound, LeftBound54488.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54492.bound, LeftBound54488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority54492.actual selector witness, LeftBound54488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54496

namespace LeftBound54500
def owner : Owner := ⟨.program ⟨214⟩, ⟨28535⟩⟩
def transferEvent : Nat := 54500
def frameStart : Nat := 54400
def rule : BoundRule := .sum [.predecessor 0 54498 .coefficient, .predecessor 1 54499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54498 .coefficient)
      LeftBound54496.bound (LeftBound54496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54499 .coefficient)
      LeftBound54477.bound (LeftBound54477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54496.bound, LeftBound54477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54496.bound, LeftBound54477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54496.actual selector witness, LeftBound54477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54500

namespace LeftBound54513
def owner : Owner := ⟨.program ⟨214⟩, ⟨28533⟩⟩
def transferEvent : Nat := 54513
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54511 .coefficient, .predecessor 1 54512 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54511 .coefficient)
      LeftBound54342.bound (LeftBound54342.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54342.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54342.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54512 .coefficient)
      LeftBound54325.bound (LeftBound54325.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54325.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54325.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54342.bound, LeftBound54325.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54342.bound, LeftBound54325.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54342.actual selector witness, LeftBound54325.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54513

namespace LeftBound54516
def owner : Owner := ⟨.program ⟨214⟩, ⟨28533⟩⟩
def transferEvent : Nat := 54516
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54510 .summary, .result 54332 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54510 .summary)
      LeftBound54344.bound (LeftBound54344.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21839⟩⟩) (rawTerms := some (Proof.Events212.exact54510RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54344.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 54332 .summary)
      LeftBound54327.bound (LeftBound54327.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28532⟩⟩) (rawTerms := some (Proof.Events212.exact54332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54327.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54344.bound, LeftBound54327.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54344.bound, LeftBound54327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54344.actual selector witness, LeftBound54327.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54516

namespace LeftBound54540
def owner : Owner := ⟨.program ⟨214⟩, ⟨11642⟩⟩
def transferEvent : Nat := 54540
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 54538 .coefficient) (.predecessor 1 54539 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54538 .coefficient)
      LeftAuthority2521.bound (LeftAuthority2521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54539 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2521.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2521.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2521.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54540

namespace LeftBound54545
def owner : Owner := ⟨.program ⟨214⟩, ⟨7275⟩⟩
def transferEvent : Nat := 54545
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54543 .coefficient) (.predecessor 1 54544 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54543 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54544 .coefficient)
      LeftBound10479.bound (LeftBound10479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events040.exact10480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10479.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound10479.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound10479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound10479.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54545

namespace LeftBound54550
def owner : Owner := ⟨.program ⟨214⟩, ⟨11643⟩⟩
def transferEvent : Nat := 54550
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54548 .coefficient, .predecessor 1 54549 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 54548 .coefficient)
      LeftBound54545.bound (LeftBound54545.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54545.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54545.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 54549 .coefficient)
      LeftBound54540.bound (LeftBound54540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events213.exact54542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54540.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54545.bound, LeftBound54540.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54545.bound, LeftBound54540.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound54545.actual selector witness, LeftBound54540.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54550

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
