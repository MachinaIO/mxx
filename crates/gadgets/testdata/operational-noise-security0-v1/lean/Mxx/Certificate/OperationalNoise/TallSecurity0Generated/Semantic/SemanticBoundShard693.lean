import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard692

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100656
def owner : Owner := ⟨.program ⟨214⟩, ⟨12265⟩⟩
def transferEvent : Nat := 100656
def frameStart : Nat := 100581
def rule : BoundRule := .sum [.predecessor 0 100654 .coefficient, .predecessor 1 100655 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100654 .coefficient)
      LeftBound100651.bound (LeftBound100651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100651.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100655 .coefficient)
      LeftBound100628.bound (LeftBound100628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100630RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100628.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100651.bound, LeftBound100628.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100651.bound, LeftBound100628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100651.actual selector witness, LeftBound100628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100656

namespace LeftBound100660
def owner : Owner := ⟨.program ⟨214⟩, ⟨25286⟩⟩
def transferEvent : Nat := 100660
def frameStart : Nat := 100581
def rule : BoundRule := .product (.predecessor 0 100658 .coefficient) (.predecessor 1 100659 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100658 .coefficient)
      LeftBound100656.bound (LeftBound100656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100657RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100656.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100659 .coefficient)
      LeftAuthority100613.bound (LeftAuthority100613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100613.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100613.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100656.bound LeftAuthority100613.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100656.bound, LeftAuthority100613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100656.actual selector witness) * (LeftAuthority100613.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100660

namespace LeftBound100671
def owner : Owner := ⟨.program ⟨214⟩, ⟨15414⟩⟩
def transferEvent : Nat := 100671
def frameStart : Nat := 100581
def rule : BoundRule := .product (.predecessor 0 100669 .coefficient) (.predecessor 1 100670 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100669 .coefficient)
      LeftAuthority100624.bound (LeftAuthority100624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100670 .coefficient)
      LeftAuthority100667.bound (LeftAuthority100667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100667.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority100624.bound LeftAuthority100667.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100624.bound, LeftAuthority100667.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority100624.actual selector witness) * (LeftAuthority100667.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100671

namespace LeftBound100679
def owner : Owner := ⟨.program ⟨214⟩, ⟨15415⟩⟩
def transferEvent : Nat := 100679
def frameStart : Nat := 100581
def rule : BoundRule := .sum [.predecessor 0 100677 .coefficient, .predecessor 1 100678 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100677 .coefficient)
      LeftAuthority100675.bound (LeftAuthority100675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100678 .coefficient)
      LeftBound100671.bound (LeftBound100671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100671.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority100675.bound, LeftBound100671.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100675.bound, LeftBound100671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority100675.actual selector witness, LeftBound100671.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100679

namespace LeftBound100683
def owner : Owner := ⟨.program ⟨214⟩, ⟨25287⟩⟩
def transferEvent : Nat := 100683
def frameStart : Nat := 100581
def rule : BoundRule := .sum [.predecessor 0 100681 .coefficient, .predecessor 1 100682 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100681 .coefficient)
      LeftBound100679.bound (LeftBound100679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100682 .coefficient)
      LeftBound100660.bound (LeftBound100660.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100660.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100660.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100679.bound, LeftBound100660.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100679.bound, LeftBound100660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100679.actual selector witness, LeftBound100660.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100683

namespace LeftBound100696
def owner : Owner := ⟨.program ⟨214⟩, ⟨25285⟩⟩
def transferEvent : Nat := 100696
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100694 .coefficient, .predecessor 1 100695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100694 .coefficient)
      LeftBound100541.bound (LeftBound100541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100695 .coefficient)
      LeftBound100524.bound (LeftBound100524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100541.bound, LeftBound100524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100541.bound, LeftBound100524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100541.actual selector witness, LeftBound100524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100696

namespace LeftBound100699
def owner : Owner := ⟨.program ⟨214⟩, ⟨25285⟩⟩
def transferEvent : Nat := 100699
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 100693 .summary, .result 100531 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100693 .summary)
      LeftBound100543.bound (LeftBound100543.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19232⟩⟩) (rawTerms := some (Proof.Events393.exact100693RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100543.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100531 .summary)
      LeftBound100526.bound (LeftBound100526.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25284⟩⟩) (rawTerms := some (Proof.Events392.exact100531RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100543.bound, LeftBound100526.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100543.bound, LeftBound100526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100543.actual selector witness, LeftBound100526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100699

namespace LeftBound100703
def owner : Owner := ⟨.program ⟨214⟩, ⟨26965⟩⟩
def transferEvent : Nat := 100703
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100701 .coefficient) (.predecessor 1 100702 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100701 .coefficient)
      LeftBound100696.bound (LeftBound100696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100702 .coefficient)
      LeftAuthority100446.bound (LeftAuthority100446.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events392.exact100447RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100446.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100696.bound LeftAuthority100446.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100696.bound, LeftAuthority100446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100696.actual selector witness) * (LeftAuthority100446.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100703

namespace LeftBound100704
def owner : Owner := ⟨.program ⟨214⟩, ⟨26965⟩⟩
def transferEvent : Nat := 100704
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26963⟩⟩]⟩ [⟨.result 100447 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100447 .coefficient)
      LeftAuthority100446.bound (LeftAuthority100446.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26963⟩⟩) (rawTerms := some (Proof.Events392.exact100447RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100446.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100446.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100446.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100446.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100446.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100704

namespace LeftBound100705
def owner : Owner := ⟨.program ⟨214⟩, ⟨26965⟩⟩
def transferEvent : Nat := 100705
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100700 .summary) (.transfer 100704) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100700 .summary)
      LeftBound100699.bound (LeftBound100699.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25285⟩⟩) (rawTerms := some (Proof.Events393.exact100700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100704)
      LeftBound100704.bound (LeftBound100704.actual selector witness) := by
  exact .transfer (LeftBound100704.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound100699.bound LeftBound100704.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100699.bound, LeftBound100704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound100699.actual selector witness) * (LeftBound100704.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100705

namespace LeftBound100716
def owner : Owner := ⟨.program ⟨214⟩, ⟨20815⟩⟩
def transferEvent : Nat := 100716
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 100714 .coefficient) (.value (.predecessor 1 100715 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100714 .coefficient)
      LeftAuthority100712.bound (LeftAuthority100712.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100712.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100715 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority100712.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100712.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100712.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100716

namespace LeftBound100720
def owner : Owner := ⟨.program ⟨214⟩, ⟨20816⟩⟩
def transferEvent : Nat := 100720
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100718 .coefficient) (.predecessor 1 100719 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100718 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100719 .coefficient)
      LeftBound100716.bound (LeftBound100716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100716.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound100716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound100716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound100716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100720

namespace LeftBound100721
def owner : Owner := ⟨.program ⟨214⟩, ⟨20816⟩⟩
def transferEvent : Nat := 100721
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20813⟩⟩]⟩ [⟨.result 100713 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100713 .coefficient)
      LeftAuthority100712.bound (LeftAuthority100712.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20813⟩⟩) (rawTerms := some (Proof.Events393.exact100713RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100712.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100712.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100712.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100712.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100721

namespace LeftBound100722
def owner : Owner := ⟨.program ⟨214⟩, ⟨20816⟩⟩
def transferEvent : Nat := 100722
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 100721) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100721)
      LeftBound100721.bound (LeftBound100721.actual selector witness) := by
  exact .transfer (LeftBound100721.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound100721.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound100721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound100721.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100722

namespace LeftBound100793
def owner : Owner := ⟨.program ⟨214⟩, ⟨15413⟩⟩
def transferEvent : Nat := 100793
def frameStart : Nat := 100766
def rule : BoundRule := .identity (.predecessor 0 100792 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100792 .coefficient)
      LeftAuthority100790.bound (LeftAuthority100790.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events393.exact100791RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100790.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100790.derived selector witness)

def rawBound : CoeffClass := LeftAuthority100790.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority100790.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound100793

namespace LeftBound100810
def owner : Owner := ⟨.program ⟨214⟩, ⟨15454⟩⟩
def transferEvent : Nat := 100810
def frameStart : Nat := 100766
def rule : BoundRule := .sum [.predecessor 0 100808 .coefficient, .predecessor 1 100809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100808 .coefficient)
      LeftBound100793.bound (LeftBound100793.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound100793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100809 .coefficient)
      LeftAuthority100806.bound (LeftAuthority100806.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority100806.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100793.bound, LeftAuthority100806.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100793.bound, LeftAuthority100806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100793.actual selector witness, LeftAuthority100806.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100810

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
