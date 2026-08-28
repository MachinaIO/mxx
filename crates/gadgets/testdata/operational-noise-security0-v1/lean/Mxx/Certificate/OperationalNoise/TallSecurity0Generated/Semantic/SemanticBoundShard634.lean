import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard605
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard633

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound93714
def owner : Owner := ⟨.program ⟨214⟩, ⟨14993⟩⟩
def transferEvent : Nat := 93714
def frameStart : Nat := 93658
def rule : BoundRule := .sum [.predecessor 0 93712 .coefficient, .predecessor 1 93713 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93712 .coefficient)
      LeftBound93697.bound (LeftBound93697.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93713 .coefficient)
      LeftAuthority93710.bound (LeftAuthority93710.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority93710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93697.bound, LeftAuthority93710.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93697.bound, LeftAuthority93710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93697.actual selector witness, LeftAuthority93710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93714

namespace LeftBound93717
def owner : Owner := ⟨.program ⟨214⟩, ⟨14994⟩⟩
def transferEvent : Nat := 93717
def frameStart : Nat := 93658
def rule : BoundRule := .identity (.predecessor 0 93716 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93716 .coefficient)
      LeftBound93714.bound (LeftBound93714.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound93714.derived selector witness)

def rawBound : CoeffClass := LeftBound93714.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound93714.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound93717

namespace LeftBound93723
def owner : Owner := ⟨.program ⟨214⟩, ⟨14995⟩⟩
def transferEvent : Nat := 93723
def frameStart : Nat := 93658
def rule : BoundRule := .product (.predecessor 0 93721 .coefficient) (.predecessor 1 93722 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93721 .coefficient)
      LeftAuthority93719.bound (LeftAuthority93719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93722 .coefficient)
      LeftBound93717.bound (LeftBound93717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority93719.bound LeftBound93717.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93719.bound, LeftBound93717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority93719.actual selector witness) * (LeftBound93717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93723

namespace LeftBound93731
def owner : Owner := ⟨.program ⟨214⟩, ⟨14996⟩⟩
def transferEvent : Nat := 93731
def frameStart : Nat := 93658
def rule : BoundRule := .sum [.predecessor 0 93729 .coefficient, .predecessor 1 93730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93729 .coefficient)
      LeftAuthority93727.bound (LeftAuthority93727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93730 .coefficient)
      LeftBound93723.bound (LeftBound93723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93723.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93727.bound, LeftBound93723.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93727.bound, LeftBound93723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93727.actual selector witness, LeftBound93723.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93731

namespace LeftBound93735
def owner : Owner := ⟨.program ⟨214⟩, ⟨26558⟩⟩
def transferEvent : Nat := 93735
def frameStart : Nat := 93658
def rule : BoundRule := .product (.predecessor 0 93733 .coefficient) (.predecessor 1 93734 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93733 .coefficient)
      LeftBound93731.bound (LeftBound93731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93734 .coefficient)
      LeftAuthority93708.bound (LeftAuthority93708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93708.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93731.bound LeftAuthority93708.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93731.bound, LeftAuthority93708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93731.actual selector witness) * (LeftAuthority93708.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93735

namespace LeftBound93746
def owner : Owner := ⟨.program ⟨214⟩, ⟨15050⟩⟩
def transferEvent : Nat := 93746
def frameStart : Nat := 93658
def rule : BoundRule := .product (.predecessor 0 93744 .coefficient) (.predecessor 1 93745 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93744 .coefficient)
      LeftAuthority93719.bound (LeftAuthority93719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93745 .coefficient)
      LeftAuthority93742.bound (LeftAuthority93742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93742.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority93719.bound LeftAuthority93742.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93719.bound, LeftAuthority93742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority93719.actual selector witness) * (LeftAuthority93742.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93746

namespace LeftBound93754
def owner : Owner := ⟨.program ⟨214⟩, ⟨15051⟩⟩
def transferEvent : Nat := 93754
def frameStart : Nat := 93658
def rule : BoundRule := .sum [.predecessor 0 93752 .coefficient, .predecessor 1 93753 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93752 .coefficient)
      LeftAuthority93750.bound (LeftAuthority93750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93753 .coefficient)
      LeftBound93746.bound (LeftBound93746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority93750.bound, LeftBound93746.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93750.bound, LeftBound93746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority93750.actual selector witness, LeftBound93746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93754

namespace LeftBound93758
def owner : Owner := ⟨.program ⟨214⟩, ⟨26563⟩⟩
def transferEvent : Nat := 93758
def frameStart : Nat := 93658
def rule : BoundRule := .sum [.predecessor 0 93756 .coefficient, .predecessor 1 93757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93756 .coefficient)
      LeftBound93754.bound (LeftBound93754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93757 .coefficient)
      LeftBound93735.bound (LeftBound93735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93754.bound, LeftBound93735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93754.bound, LeftBound93735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93754.actual selector witness, LeftBound93735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93758

namespace LeftBound93771
def owner : Owner := ⟨.program ⟨214⟩, ⟨26560⟩⟩
def transferEvent : Nat := 93771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 93769 .coefficient, .predecessor 1 93770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93769 .coefficient)
      LeftBound93600.bound (LeftBound93600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93770 .coefficient)
      LeftBound93583.bound (LeftBound93583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events365.exact93590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93583.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93600.bound, LeftBound93583.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93600.bound, LeftBound93583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93600.actual selector witness, LeftBound93583.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93771

namespace LeftBound93774
def owner : Owner := ⟨.program ⟨214⟩, ⟨26560⟩⟩
def transferEvent : Nat := 93774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 93768 .summary, .result 93590 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93768 .summary)
      LeftBound93602.bound (LeftBound93602.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20467⟩⟩) (rawTerms := some (Proof.Events366.exact93768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93590 .summary)
      LeftBound93585.bound (LeftBound93585.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26559⟩⟩) (rawTerms := some (Proof.Events365.exact93590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound93602.bound, LeftBound93585.bound]
def bound : CoeffClass := .finite ⟨1291900380601931935744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93602.bound, LeftBound93585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound93602.actual selector witness, LeftBound93585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound93774

namespace LeftBound93778
def owner : Owner := ⟨.program ⟨214⟩, ⟨26561⟩⟩
def transferEvent : Nat := 93778
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93776 .coefficient) (.predecessor 1 93777 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93776 .coefficient)
      LeftBound93771.bound (LeftBound93771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound93771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound93771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93777 .coefficient)
      LeftBound5838.bound (LeftBound5838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93771.bound LeftBound5838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93771.bound, LeftBound5838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93771.actual selector witness) * (LeftBound5838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93778

namespace LeftBound93779
def owner : Owner := ⟨.program ⟨214⟩, ⟨26561⟩⟩
def transferEvent : Nat := 93779
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6671⟩⟩]⟩ [⟨.result 5835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5835 .coefficient)
      LeftAuthority5834.bound (LeftAuthority5834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6671⟩⟩) (rawTerms := some (Proof.Events022.exact5835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5834.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93779

namespace LeftBound93780
def owner : Owner := ⟨.program ⟨214⟩, ⟨26561⟩⟩
def transferEvent : Nat := 93780
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 93775 .summary) (.transfer 93779) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93775 .summary)
      LeftBound93774.bound (LeftBound93774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26560⟩⟩) (rawTerms := some (Proof.Events366.exact93775RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound93774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93779)
      LeftBound93779.bound (LeftBound93779.actual selector witness) := by
  exact .transfer (LeftBound93779.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound93774.bound LeftBound93779.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound93774.bound, LeftBound93779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound93774.actual selector witness) * (LeftBound93779.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93780

namespace LeftBound93795
def owner : Owner := ⟨.program ⟨214⟩, ⟨26353⟩⟩
def transferEvent : Nat := 93795
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 93793 .coefficient) (.predecessor 1 93794 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 93793 .coefficient)
      LeftBound88352.bound (LeftBound88352.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events345.exact88356RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound88352.bound, RecordedBoundRefines] <;> decide)
      (LeftBound88352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 93794 .coefficient)
      LeftAuthority93791.bound (LeftAuthority93791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events366.exact93792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93791.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88352.bound LeftAuthority93791.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88352.bound, LeftAuthority93791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88352.actual selector witness) * (LeftAuthority93791.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93795

namespace LeftBound93796
def owner : Owner := ⟨.program ⟨214⟩, ⟨26353⟩⟩
def transferEvent : Nat := 93796
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26351⟩⟩]⟩ [⟨.result 93792 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 93792 .coefficient)
      LeftAuthority93791.bound (LeftAuthority93791.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26351⟩⟩) (rawTerms := some (Proof.Events366.exact93792RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority93791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority93791.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority93791.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority93791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority93791.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound93796

namespace LeftBound93797
def owner : Owner := ⟨.program ⟨214⟩, ⟨26353⟩⟩
def transferEvent : Nat := 93797
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 88356 .summary) (.transfer 93796) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 88356 .summary)
      LeftBound88355.bound (LeftBound88355.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24913⟩⟩) (rawTerms := some (Proof.Events345.exact88356RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound88355.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 93796)
      LeftBound93796.bound (LeftBound93796.actual selector witness) := by
  exact .transfer (LeftBound93796.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound88355.bound LeftBound93796.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound88355.bound, LeftBound93796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound88355.actual selector witness) * (LeftBound93796.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound93797

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
