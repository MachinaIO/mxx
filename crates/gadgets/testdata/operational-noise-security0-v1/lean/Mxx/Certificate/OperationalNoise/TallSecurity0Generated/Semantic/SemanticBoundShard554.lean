import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard553

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81456
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def transferEvent : Nat := 81456
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20032⟩⟩]⟩ [⟨.result 81448 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81448 .coefficient)
      LeftAuthority81447.bound (LeftAuthority81447.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20032⟩⟩) (rawTerms := some (Proof.Events318.exact81448RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81447.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81447.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81447.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81447.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81456

namespace LeftBound81457
def owner : Owner := ⟨.program ⟨214⟩, ⟨20035⟩⟩
def transferEvent : Nat := 81457
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 81456) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81456)
      LeftBound81456.bound (LeftBound81456.actual selector witness) := by
  exact .transfer (LeftBound81456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound81456.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound81456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound81456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81457

namespace LeftBound81536
def owner : Owner := ⟨.program ⟨214⟩, ⟨12763⟩⟩
def transferEvent : Nat := 81536
def frameStart : Nat := 81507
def rule : BoundRule := .product (.predecessor 0 81534 .coefficient) (.predecessor 1 81535 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81534 .coefficient)
      LeftAuthority81532.bound (LeftAuthority81532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81535 .coefficient)
      LeftAuthority81529.bound (LeftAuthority81529.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81529.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81529.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81532.bound LeftAuthority81529.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81532.bound, LeftAuthority81529.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81532.actual selector witness) * (LeftAuthority81529.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81536

namespace LeftBound81540
def owner : Owner := ⟨.program ⟨214⟩, ⟨12764⟩⟩
def transferEvent : Nat := 81540
def frameStart : Nat := 81507
def rule : BoundRule := .identity (.predecessor 0 81539 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81539 .coefficient)
      LeftBound81536.bound (LeftBound81536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81536.derived selector witness)

def rawBound : CoeffClass := LeftBound81536.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81536.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81540

namespace LeftBound81557
def owner : Owner := ⟨.program ⟨214⟩, ⟨12858⟩⟩
def transferEvent : Nat := 81557
def frameStart : Nat := 81507
def rule : BoundRule := .sum [.predecessor 0 81555 .coefficient, .predecessor 1 81556 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81555 .coefficient)
      LeftBound81540.bound (LeftBound81540.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81556 .coefficient)
      LeftAuthority81553.bound (LeftAuthority81553.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81540.bound, LeftAuthority81553.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81540.bound, LeftAuthority81553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81540.actual selector witness, LeftAuthority81553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81557

namespace LeftBound81560
def owner : Owner := ⟨.program ⟨214⟩, ⟨12859⟩⟩
def transferEvent : Nat := 81560
def frameStart : Nat := 81507
def rule : BoundRule := .identity (.predecessor 0 81559 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81559 .coefficient)
      LeftBound81557.bound (LeftBound81557.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81557.derived selector witness)

def rawBound : CoeffClass := LeftBound81557.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81557.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81560

namespace LeftBound81566
def owner : Owner := ⟨.program ⟨214⟩, ⟨12860⟩⟩
def transferEvent : Nat := 81566
def frameStart : Nat := 81507
def rule : BoundRule := .product (.predecessor 0 81564 .coefficient) (.predecessor 1 81565 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81564 .coefficient)
      LeftAuthority81562.bound (LeftAuthority81562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81565 .coefficient)
      LeftBound81560.bound (LeftBound81560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81560.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81560.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority81562.bound LeftBound81560.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81562.bound, LeftBound81560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority81562.actual selector witness) * (LeftBound81560.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81566

namespace LeftBound81580
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 81580
def frameStart : Nat := 81507
def rule : BoundRule := .scale (.predecessor 0 81578 .coefficient) (.value (.predecessor 1 81579 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81578 .coefficient)
      LeftAuthority81576.bound (LeftAuthority81576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81577RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81576.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81579 .coefficient)
      LeftAuthority81510.bound (LeftAuthority81510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81510.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81576.bound LeftAuthority81510.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81576.bound, LeftAuthority81510.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81576.actual selector witness) * (LeftAuthority81510.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81580

namespace LeftBound81583
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 81583
def frameStart : Nat := 81507
def rule : BoundRule := .identity (.predecessor 0 81582 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81582 .coefficient)
      LeftAuthority81570.bound (LeftAuthority81570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81570.derived selector witness)

def rawBound : CoeffClass := LeftAuthority81570.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81570.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority81570.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81583

namespace LeftBound81587
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 81587
def frameStart : Nat := 81507
def rule : BoundRule := .product (.predecessor 0 81585 .coefficient) (.predecessor 1 81586 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81585 .coefficient)
      LeftBound81583.bound (LeftBound81583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81586 .coefficient)
      LeftBound81580.bound (LeftBound81580.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81580.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81580.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81583.bound LeftBound81580.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81583.bound, LeftBound81580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81583.actual selector witness) * (LeftBound81580.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81587

namespace LeftBound81592
def owner : Owner := ⟨.program ⟨214⟩, ⟨12861⟩⟩
def transferEvent : Nat := 81592
def frameStart : Nat := 81507
def rule : BoundRule := .sum [.predecessor 0 81590 .coefficient, .predecessor 1 81591 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81590 .coefficient)
      LeftBound81587.bound (LeftBound81587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81591 .coefficient)
      LeftBound81566.bound (LeftBound81566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81587.bound, LeftBound81566.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81587.bound, LeftBound81566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81587.actual selector witness, LeftBound81566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81592

namespace LeftBound81596
def owner : Owner := ⟨.program ⟨214⟩, ⟨25530⟩⟩
def transferEvent : Nat := 81596
def frameStart : Nat := 81507
def rule : BoundRule := .product (.predecessor 0 81594 .coefficient) (.predecessor 1 81595 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81594 .coefficient)
      LeftBound81592.bound (LeftBound81592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81595 .coefficient)
      LeftAuthority81551.bound (LeftAuthority81551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81551.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81551.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81592.bound LeftAuthority81551.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81592.bound, LeftAuthority81551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81592.actual selector witness) * (LeftAuthority81551.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81596

namespace LeftBound81607
def owner : Owner := ⟨.program ⟨214⟩, ⟨16635⟩⟩
def transferEvent : Nat := 81607
def frameStart : Nat := 81507
def rule : BoundRule := .product (.predecessor 0 81605 .coefficient) (.predecessor 1 81606 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81605 .coefficient)
      LeftAuthority81562.bound (LeftAuthority81562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81606 .coefficient)
      LeftAuthority81603.bound (LeftAuthority81603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81604RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81603.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81603.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81562.bound LeftAuthority81603.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81562.bound, LeftAuthority81603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81562.actual selector witness) * (LeftAuthority81603.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81607

namespace LeftBound81615
def owner : Owner := ⟨.program ⟨214⟩, ⟨16636⟩⟩
def transferEvent : Nat := 81615
def frameStart : Nat := 81507
def rule : BoundRule := .sum [.predecessor 0 81613 .coefficient, .predecessor 1 81614 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81613 .coefficient)
      LeftAuthority81611.bound (LeftAuthority81611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81612RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81611.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81611.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81614 .coefficient)
      LeftBound81607.bound (LeftBound81607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81607.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81607.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81611.bound, LeftBound81607.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81611.bound, LeftBound81607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81611.actual selector witness, LeftBound81607.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81615

namespace LeftBound81619
def owner : Owner := ⟨.program ⟨214⟩, ⟨25531⟩⟩
def transferEvent : Nat := 81619
def frameStart : Nat := 81507
def rule : BoundRule := .sum [.predecessor 0 81617 .coefficient, .predecessor 1 81618 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81617 .coefficient)
      LeftBound81615.bound (LeftBound81615.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81615.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81615.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81618 .coefficient)
      LeftBound81596.bound (LeftBound81596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81601RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81615.bound, LeftBound81596.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81615.bound, LeftBound81596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81615.actual selector witness, LeftBound81596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81619

namespace LeftBound81632
def owner : Owner := ⟨.program ⟨214⟩, ⟨25529⟩⟩
def transferEvent : Nat := 81632
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 81630 .coefficient, .predecessor 1 81631 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81630 .coefficient)
      LeftBound81455.bound (LeftBound81455.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81455.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81455.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81631 .coefficient)
      LeftBound81438.bound (LeftBound81438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81438.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81455.bound, LeftBound81438.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81455.bound, LeftBound81438.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81455.actual selector witness, LeftBound81438.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81632

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
