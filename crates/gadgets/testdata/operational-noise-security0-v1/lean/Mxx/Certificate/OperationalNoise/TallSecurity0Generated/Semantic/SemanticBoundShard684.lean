import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard683

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99612
def owner : Owner := ⟨.program ⟨214⟩, ⟨13749⟩⟩
def transferEvent : Nat := 99612
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99610 .coefficient) (.predecessor 1 99611 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99610 .coefficient)
      LeftBound99606.bound (LeftBound99606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99606.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99611 .coefficient)
      LeftAuthority4844.bound (LeftAuthority4844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound99606.bound LeftAuthority4844.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99606.bound, LeftAuthority4844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound99606.actual selector witness) * (LeftAuthority4844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99612

namespace LeftBound99613
def owner : Owner := ⟨.program ⟨214⟩, ⟨13749⟩⟩
def transferEvent : Nat := 99613
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13746⟩⟩], []⟩ [⟨.result 4845 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4845 .coefficient)
      LeftAuthority4844.bound (LeftAuthority4844.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13746⟩⟩) (rawTerms := some (Proof.Events018.exact4845RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4844.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4844.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4844.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99613

namespace LeftBound99614
def owner : Owner := ⟨.program ⟨214⟩, ⟨13749⟩⟩
def transferEvent : Nat := 99614
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99609 .summary) (.transfer 99613) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99609 .summary)
      LeftBound99607.bound (LeftBound99607.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11292⟩⟩) (rawTerms := some (Proof.Events389.exact99609RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99607.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99613)
      LeftBound99613.bound (LeftBound99613.actual selector witness) := by
  exact .transfer (LeftBound99613.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound99607.bound LeftBound99613.bound
def bound : CoeffClass := .finite ⟨9984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99607.bound, LeftBound99613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound99607.actual selector witness) * (LeftBound99613.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99614

namespace LeftBound99620
def owner : Owner := ⟨.program ⟨214⟩, ⟨13750⟩⟩
def transferEvent : Nat := 99620
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 99618 .coefficient) (.predecessor 1 99619 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99618 .coefficient)
      LeftAuthority4844.bound (LeftAuthority4844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events018.exact4845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99619 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4844.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4844.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4844.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound99620

namespace LeftBound99625
def owner : Owner := ⟨.program ⟨214⟩, ⟨7131⟩⟩
def transferEvent : Nat := 99625
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99623 .coefficient) (.predecessor 1 99624 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99623 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99624 .coefficient)
      LeftBound12524.bound (LeftBound12524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound12524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound12524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound12524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99625

namespace LeftBound99630
def owner : Owner := ⟨.program ⟨214⟩, ⟨13751⟩⟩
def transferEvent : Nat := 99630
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99628 .coefficient, .predecessor 1 99629 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99628 .coefficient)
      LeftBound99625.bound (LeftBound99625.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99627RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99625.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99625.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99629 .coefficient)
      LeftBound99620.bound (LeftBound99620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99620.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99625.bound, LeftBound99620.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99625.bound, LeftBound99620.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99625.actual selector witness, LeftBound99620.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99630

namespace LeftBound99634
def owner : Owner := ⟨.program ⟨214⟩, ⟨13752⟩⟩
def transferEvent : Nat := 99634
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99632 .coefficient, .predecessor 1 99633 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99632 .coefficient)
      LeftBound99630.bound (LeftBound99630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99630.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99630.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99633 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99630.bound, LeftBound12516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99630.bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99630.actual selector witness, LeftBound12516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99634

namespace LeftBound99635
def owner : Owner := ⟨.program ⟨214⟩, ⟨13752⟩⟩
def transferEvent : Nat := 99635
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨108⟩⟩]⟩ [⟨.result 12517 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12517 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨108⟩⟩) (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12516.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12516.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99635

namespace LeftBound99640
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def transferEvent : Nat := 99640
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99638 .coefficient) (.predecessor 1 99639 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99638 .coefficient)
      LeftBound99634.bound (LeftBound99634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99634.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99634.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99639 .coefficient)
      LeftBound12513.bound (LeftBound12513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12514RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12513.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99634.bound LeftBound12513.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99634.bound, LeftBound12513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99634.actual selector witness) * (LeftBound12513.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99640

namespace LeftBound99641
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def transferEvent : Nat := 99641
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7846⟩⟩]⟩ [⟨.result 12510 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12510 .coefficient)
      LeftAuthority12509.bound (LeftAuthority12509.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7846⟩⟩) (rawTerms := some (Proof.Events048.exact12510RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12509.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12509.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12509.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12509.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99641

namespace LeftBound99642
def owner : Owner := ⟨.program ⟨214⟩, ⟨13753⟩⟩
def transferEvent : Nat := 99642
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99637 .summary) (.transfer 99641) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99637 .summary)
      LeftBound99635.bound (LeftBound99635.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13752⟩⟩) (rawTerms := some (Proof.Events389.exact99637RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99641)
      LeftBound99641.bound (LeftBound99641.actual selector witness) := by
  exact .transfer (LeftBound99641.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99635.bound LeftBound99641.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99635.bound, LeftBound99641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99635.actual selector witness) * (LeftBound99641.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99642

namespace LeftBound99650
def owner : Owner := ⟨.program ⟨214⟩, ⟨13754⟩⟩
def transferEvent : Nat := 99650
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99648 .coefficient, .predecessor 1 99649 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99648 .coefficient)
      LeftBound99640.bound (LeftBound99640.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99647RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99640.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99640.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99649 .coefficient)
      LeftBound99612.bound (LeftBound99612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99640.bound, LeftBound99612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99640.bound, LeftBound99612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99640.actual selector witness, LeftBound99612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99650

namespace LeftBound99652
def owner : Owner := ⟨.program ⟨214⟩, ⟨13754⟩⟩
def transferEvent : Nat := 99652
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99647 .summary, .result 99617 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99647 .summary)
      LeftBound99642.bound (LeftBound99642.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13753⟩⟩) (rawTerms := some (Proof.Events389.exact99647RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99617 .summary)
      LeftBound99614.bound (LeftBound99614.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13749⟩⟩) (rawTerms := some (Proof.Events389.exact99617RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99642.bound, LeftBound99614.bound]
def bound : CoeffClass := .finite ⟨95430400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99642.bound, LeftBound99614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99642.actual selector witness, LeftBound99614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99652

namespace LeftBound99656
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def transferEvent : Nat := 99656
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 99654 .coefficient) (.predecessor 1 99655 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99654 .coefficient)
      LeftBound99650.bound (LeftBound99650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99650.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99655 .coefficient)
      LeftAuthority99588.bound (LeftAuthority99588.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events389.exact99589RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99588.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99650.bound LeftAuthority99588.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99650.bound, LeftAuthority99588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99650.actual selector witness) * (LeftAuthority99588.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99656

namespace LeftBound99657
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def transferEvent : Nat := 99657
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25899⟩⟩]⟩ [⟨.result 99589 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99589 .coefficient)
      LeftAuthority99588.bound (LeftAuthority99588.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25899⟩⟩) (rawTerms := some (Proof.Events389.exact99589RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99588.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99588.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority99588.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99588.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority99588.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound99657

namespace LeftBound99658
def owner : Owner := ⟨.program ⟨214⟩, ⟨25900⟩⟩
def transferEvent : Nat := 99658
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 99653 .summary) (.transfer 99657) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99653 .summary)
      LeftBound99652.bound (LeftBound99652.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13754⟩⟩) (rawTerms := some (Proof.Events389.exact99653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 99657)
      LeftBound99657.bound (LeftBound99657.actual selector witness) := by
  exact .transfer (LeftBound99657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound99652.bound LeftBound99657.bound
def bound : CoeffClass := .finite ⟨350231094886400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99652.bound, LeftBound99657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound99652.actual selector witness) * (LeftBound99657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99658

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
