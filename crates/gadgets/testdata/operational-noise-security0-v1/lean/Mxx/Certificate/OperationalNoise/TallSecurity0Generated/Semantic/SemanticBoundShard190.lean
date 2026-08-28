import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard087
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard088
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard189

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28612
def owner : Owner := ⟨.program ⟨214⟩, ⟨17361⟩⟩
def transferEvent : Nat := 28612
def frameStart : Nat := 28524
def rule : BoundRule := .product (.predecessor 0 28610 .coefficient) (.predecessor 1 28611 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28610 .coefficient)
      LeftAuthority28585.bound (LeftAuthority28585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28611 .coefficient)
      LeftAuthority28608.bound (LeftAuthority28608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28608.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority28585.bound LeftAuthority28608.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28585.bound, LeftAuthority28608.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority28585.actual selector witness) * (LeftAuthority28608.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28612

namespace LeftBound28620
def owner : Owner := ⟨.program ⟨214⟩, ⟨17362⟩⟩
def transferEvent : Nat := 28620
def frameStart : Nat := 28524
def rule : BoundRule := .sum [.predecessor 0 28618 .coefficient, .predecessor 1 28619 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28618 .coefficient)
      LeftAuthority28616.bound (LeftAuthority28616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28617RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28616.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28619 .coefficient)
      LeftBound28612.bound (LeftBound28612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28612.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28612.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28616.bound, LeftBound28612.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28616.bound, LeftBound28612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28616.actual selector witness, LeftBound28612.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28620

namespace LeftBound28624
def owner : Owner := ⟨.program ⟨214⟩, ⟨27042⟩⟩
def transferEvent : Nat := 28624
def frameStart : Nat := 28524
def rule : BoundRule := .sum [.predecessor 0 28622 .coefficient, .predecessor 1 28623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28622 .coefficient)
      LeftBound28620.bound (LeftBound28620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28623 .coefficient)
      LeftBound28601.bound (LeftBound28601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28606RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28601.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28620.bound, LeftBound28601.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28620.bound, LeftBound28601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28620.actual selector witness, LeftBound28601.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28624

namespace LeftBound28637
def owner : Owner := ⟨.program ⟨214⟩, ⟨27040⟩⟩
def transferEvent : Nat := 28637
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28635 .coefficient, .predecessor 1 28636 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28635 .coefficient)
      LeftBound28466.bound (LeftBound28466.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28634RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28466.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28636 .coefficient)
      LeftBound28449.bound (LeftBound28449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28456RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28449.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28449.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28466.bound, LeftBound28449.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28466.bound, LeftBound28449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28466.actual selector witness, LeftBound28449.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28637

namespace LeftBound28640
def owner : Owner := ⟨.program ⟨214⟩, ⟨27040⟩⟩
def transferEvent : Nat := 28640
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28634 .summary, .result 28456 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28634 .summary)
      LeftBound28468.bound (LeftBound28468.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20839⟩⟩) (rawTerms := some (Proof.Events111.exact28634RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28456 .summary)
      LeftBound28451.bound (LeftBound28451.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27039⟩⟩) (rawTerms := some (Proof.Events111.exact28456RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28451.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28468.bound, LeftBound28451.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28468.bound, LeftBound28451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28468.actual selector witness, LeftBound28451.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28640

namespace LeftBound28664
def owner : Owner := ⟨.program ⟨214⟩, ⟨11004⟩⟩
def transferEvent : Nat := 28664
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 28662 .coefficient) (.predecessor 1 28663 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28662 .coefficient)
      LeftAuthority1186.bound (LeftAuthority1186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1186.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28663 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1186.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1186.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1186.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28664

namespace LeftBound28669
def owner : Owner := ⟨.program ⟨214⟩, ⟨7344⟩⟩
def transferEvent : Nat := 28669
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28667 .coefficient) (.predecessor 1 28668 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28667 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28668 .coefficient)
      LeftBound13986.bound (LeftBound13986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13986.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound13986.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound13986.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound13986.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28669

namespace LeftBound28674
def owner : Owner := ⟨.program ⟨214⟩, ⟨11005⟩⟩
def transferEvent : Nat := 28674
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28672 .coefficient, .predecessor 1 28673 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28672 .coefficient)
      LeftBound28669.bound (LeftBound28669.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28671RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28669.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28669.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28673 .coefficient)
      LeftBound28664.bound (LeftBound28664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28666RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28664.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28669.bound, LeftBound28664.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28669.bound, LeftBound28664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28669.actual selector witness, LeftBound28664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28674

namespace LeftBound28678
def owner : Owner := ⟨.program ⟨214⟩, ⟨11006⟩⟩
def transferEvent : Nat := 28678
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28676 .coefficient, .predecessor 1 28677 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28676 .coefficient)
      LeftBound28674.bound (LeftBound28674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28674.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28677 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28674.bound, LeftBound13978.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28674.bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28674.actual selector witness, LeftBound13978.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28678

namespace LeftBound28679
def owner : Owner := ⟨.program ⟨214⟩, ⟨11006⟩⟩
def transferEvent : Nat := 28679
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨88⟩⟩]⟩ [⟨.result 13979 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13979 .coefficient)
      LeftBound13978.bound (LeftBound13978.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨88⟩⟩) (rawTerms := some (Proof.Events054.exact13979RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13978.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13978.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13978.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28679

namespace LeftBound28684
def owner : Owner := ⟨.program ⟨214⟩, ⟨11007⟩⟩
def transferEvent : Nat := 28684
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28682 .coefficient) (.predecessor 1 28683 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28682 .coefficient)
      LeftBound28678.bound (LeftBound28678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28683 .coefficient)
      LeftAuthority1189.bound (LeftAuthority1189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1189.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound28678.bound LeftAuthority1189.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28678.bound, LeftAuthority1189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound28678.actual selector witness) * (LeftAuthority1189.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28684

namespace LeftBound28685
def owner : Owner := ⟨.program ⟨214⟩, ⟨11007⟩⟩
def transferEvent : Nat := 28685
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨10857⟩⟩], []⟩ [⟨.result 1190 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1190 .coefficient)
      LeftAuthority1189.bound (LeftAuthority1189.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨10857⟩⟩) (rawTerms := some (Proof.Events004.exact1190RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1189.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1189.bound []
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1189.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28685

namespace LeftBound28686
def owner : Owner := ⟨.program ⟨214⟩, ⟨11007⟩⟩
def transferEvent : Nat := 28686
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28681 .summary) (.transfer 28685) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28681 .summary)
      LeftBound28679.bound (LeftBound28679.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11006⟩⟩) (rawTerms := some (Proof.Events112.exact28681RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28679.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28685)
      LeftBound28685.bound (LeftBound28685.actual selector witness) := by
  exact .transfer (LeftBound28685.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound28679.bound LeftBound28685.bound
def bound : CoeffClass := .finite ⟨3328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28679.bound, LeftBound28685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound28679.actual selector witness) * (LeftBound28685.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28686

namespace LeftBound28692
def owner : Owner := ⟨.program ⟨214⟩, ⟨10858⟩⟩
def transferEvent : Nat := 28692
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 28690 .coefficient) (.predecessor 1 28691 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28690 .coefficient)
      LeftAuthority1189.bound (LeftAuthority1189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28691 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1189.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1189.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1189.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28692

namespace LeftBound28697
def owner : Owner := ⟨.program ⟨214⟩, ⟨7361⟩⟩
def transferEvent : Nat := 28697
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28695 .coefficient) (.predecessor 1 28696 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28695 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28696 .coefficient)
      LeftBound14027.bound (LeftBound14027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events054.exact14028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound14027.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound14027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound14027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28697

namespace LeftBound28702
def owner : Owner := ⟨.program ⟨214⟩, ⟨10859⟩⟩
def transferEvent : Nat := 28702
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28700 .coefficient, .predecessor 1 28701 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28700 .coefficient)
      LeftBound28697.bound (LeftBound28697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28701 .coefficient)
      LeftBound28692.bound (LeftBound28692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events112.exact28694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28692.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28697.bound, LeftBound28692.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28697.bound, LeftBound28692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28697.actual selector witness, LeftBound28692.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28702

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
