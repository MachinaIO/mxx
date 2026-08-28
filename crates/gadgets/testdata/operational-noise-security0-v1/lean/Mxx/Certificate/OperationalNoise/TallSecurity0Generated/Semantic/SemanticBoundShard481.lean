import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard480

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound70660
def owner : Owner := ⟨.program ⟨214⟩, ⟨13988⟩⟩
def transferEvent : Nat := 70660
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7849⟩⟩]⟩ [⟨.result 12009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12009 .coefficient)
      LeftAuthority12008.bound (LeftAuthority12008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7849⟩⟩) (rawTerms := some (Proof.Events046.exact12009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12008.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70660

namespace LeftBound70661
def owner : Owner := ⟨.program ⟨214⟩, ⟨13988⟩⟩
def transferEvent : Nat := 70661
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70656 .summary) (.transfer 70660) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70656 .summary)
      LeftBound70654.bound (LeftBound70654.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13987⟩⟩) (rawTerms := some (Proof.Events276.exact70656RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70654.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70660)
      LeftBound70660.bound (LeftBound70660.actual selector witness) := by
  exact .transfer (LeftBound70660.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70654.bound LeftBound70660.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70654.bound, LeftBound70660.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70654.actual selector witness) * (LeftBound70660.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70661

namespace LeftBound70669
def owner : Owner := ⟨.program ⟨214⟩, ⟨13989⟩⟩
def transferEvent : Nat := 70669
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 70667 .coefficient, .predecessor 1 70668 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70667 .coefficient)
      LeftBound70659.bound (LeftBound70659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70668 .coefficient)
      LeftBound70631.bound (LeftBound70631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70659.bound, LeftBound70631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70659.bound, LeftBound70631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70659.actual selector witness, LeftBound70631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70669

namespace LeftBound70671
def owner : Owner := ⟨.program ⟨214⟩, ⟨13989⟩⟩
def transferEvent : Nat := 70671
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 70666 .summary, .result 70636 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70666 .summary)
      LeftBound70661.bound (LeftBound70661.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13988⟩⟩) (rawTerms := some (Proof.Events276.exact70666RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70636 .summary)
      LeftBound70633.bound (LeftBound70633.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13984⟩⟩) (rawTerms := some (Proof.Events275.exact70636RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70633.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70661.bound, LeftBound70633.bound]
def bound : CoeffClass := .finite ⟨95433728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70661.bound, LeftBound70633.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70661.actual selector witness, LeftBound70633.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70671

namespace LeftBound70675
def owner : Owner := ⟨.program ⟨214⟩, ⟨25985⟩⟩
def transferEvent : Nat := 70675
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70673 .coefficient) (.predecessor 1 70674 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70673 .coefficient)
      LeftBound70669.bound (LeftBound70669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70674 .coefficient)
      LeftAuthority70607.bound (LeftAuthority70607.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events275.exact70608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70607.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70669.bound LeftAuthority70607.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70669.bound, LeftAuthority70607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70669.actual selector witness) * (LeftAuthority70607.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70675

namespace LeftBound70676
def owner : Owner := ⟨.program ⟨214⟩, ⟨25985⟩⟩
def transferEvent : Nat := 70676
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25984⟩⟩]⟩ [⟨.result 70608 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70608 .coefficient)
      LeftAuthority70607.bound (LeftAuthority70607.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25984⟩⟩) (rawTerms := some (Proof.Events275.exact70608RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70607.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70607.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70607.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70607.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70676

namespace LeftBound70677
def owner : Owner := ⟨.program ⟨214⟩, ⟨25985⟩⟩
def transferEvent : Nat := 70677
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 70672 .summary) (.transfer 70676) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70672 .summary)
      LeftBound70671.bound (LeftBound70671.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13989⟩⟩) (rawTerms := some (Proof.Events276.exact70672RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound70671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70676)
      LeftBound70676.bound (LeftBound70676.actual selector witness) := by
  exact .transfer (LeftBound70676.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound70671.bound LeftBound70676.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70671.bound, LeftBound70676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound70671.actual selector witness) * (LeftBound70676.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70677

namespace LeftBound70688
def owner : Owner := ⟨.program ⟨214⟩, ⟨19454⟩⟩
def transferEvent : Nat := 70688
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 70686 .coefficient) (.value (.predecessor 1 70687 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70686 .coefficient)
      LeftAuthority70684.bound (LeftAuthority70684.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70685RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70684.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70687 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority70684.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70684.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70684.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound70688

namespace LeftBound70692
def owner : Owner := ⟨.program ⟨214⟩, ⟨19455⟩⟩
def transferEvent : Nat := 70692
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 70690 .coefficient) (.predecessor 1 70691 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70690 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70691 .coefficient)
      LeftBound70688.bound (LeftBound70688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70688.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70688.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound70688.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound70688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound70688.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70692

namespace LeftBound70693
def owner : Owner := ⟨.program ⟨214⟩, ⟨19455⟩⟩
def transferEvent : Nat := 70693
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19452⟩⟩]⟩ [⟨.result 70685 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 70685 .coefficient)
      LeftAuthority70684.bound (LeftAuthority70684.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19452⟩⟩) (rawTerms := some (Proof.Events276.exact70685RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70684.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70684.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority70684.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70684.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority70684.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound70693

namespace LeftBound70694
def owner : Owner := ⟨.program ⟨214⟩, ⟨19455⟩⟩
def transferEvent : Nat := 70694
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 70693) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 70693)
      LeftBound70693.bound (LeftBound70693.actual selector witness) := by
  exact .transfer (LeftBound70693.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound70693.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound70693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound70693.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70694

namespace LeftBound70773
def owner : Owner := ⟨.program ⟨214⟩, ⟨13982⟩⟩
def transferEvent : Nat := 70773
def frameStart : Nat := 70744
def rule : BoundRule := .product (.predecessor 0 70771 .coefficient) (.predecessor 1 70772 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70771 .coefficient)
      LeftAuthority70769.bound (LeftAuthority70769.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70770RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70769.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70769.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70772 .coefficient)
      LeftAuthority70766.bound (LeftAuthority70766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority70769.bound LeftAuthority70766.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70769.bound, LeftAuthority70766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority70769.actual selector witness) * (LeftAuthority70766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70773

namespace LeftBound70777
def owner : Owner := ⟨.program ⟨214⟩, ⟨13983⟩⟩
def transferEvent : Nat := 70777
def frameStart : Nat := 70744
def rule : BoundRule := .identity (.predecessor 0 70776 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70776 .coefficient)
      LeftBound70773.bound (LeftBound70773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70773.derived selector witness)

def rawBound : CoeffClass := LeftBound70773.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70773.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70777

namespace LeftBound70794
def owner : Owner := ⟨.program ⟨214⟩, ⟨14093⟩⟩
def transferEvent : Nat := 70794
def frameStart : Nat := 70744
def rule : BoundRule := .sum [.predecessor 0 70792 .coefficient, .predecessor 1 70793 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70792 .coefficient)
      LeftBound70777.bound (LeftBound70777.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70793 .coefficient)
      LeftAuthority70790.bound (LeftAuthority70790.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority70790.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound70777.bound, LeftAuthority70790.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70777.bound, LeftAuthority70790.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound70777.actual selector witness, LeftAuthority70790.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound70794

namespace LeftBound70797
def owner : Owner := ⟨.program ⟨214⟩, ⟨14094⟩⟩
def transferEvent : Nat := 70797
def frameStart : Nat := 70744
def rule : BoundRule := .identity (.predecessor 0 70796 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70796 .coefficient)
      LeftBound70794.bound (LeftBound70794.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound70794.derived selector witness)

def rawBound : CoeffClass := LeftBound70794.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound70794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound70794.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound70797

namespace LeftBound70803
def owner : Owner := ⟨.program ⟨214⟩, ⟨14095⟩⟩
def transferEvent : Nat := 70803
def frameStart : Nat := 70744
def rule : BoundRule := .product (.predecessor 0 70801 .coefficient) (.predecessor 1 70802 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 70801 .coefficient)
      LeftAuthority70799.bound (LeftAuthority70799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70800RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority70799.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority70799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 70802 .coefficient)
      LeftBound70797.bound (LeftBound70797.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events276.exact70798RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound70797.bound, RecordedBoundRefines] <;> decide)
      (LeftBound70797.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority70799.bound LeftBound70797.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority70799.bound, LeftBound70797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority70799.actual selector witness) * (LeftBound70797.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound70803

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
