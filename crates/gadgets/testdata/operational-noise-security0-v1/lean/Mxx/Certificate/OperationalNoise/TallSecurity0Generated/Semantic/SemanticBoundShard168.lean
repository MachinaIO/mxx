import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard064
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard167

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound25691
def owner : Owner := ⟨.program ⟨214⟩, ⟨16231⟩⟩
def transferEvent : Nat := 25691
def frameStart : Nat := 25632
def rule : BoundRule := .identity (.predecessor 0 25690 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25690 .coefficient)
      LeftBound25688.bound (LeftBound25688.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound25688.derived selector witness)

def rawBound : CoeffClass := LeftBound25688.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25688.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound25688.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound25691

namespace LeftBound25697
def owner : Owner := ⟨.program ⟨214⟩, ⟨16232⟩⟩
def transferEvent : Nat := 25697
def frameStart : Nat := 25632
def rule : BoundRule := .product (.predecessor 0 25695 .coefficient) (.predecessor 1 25696 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25695 .coefficient)
      LeftAuthority25693.bound (LeftAuthority25693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25696 .coefficient)
      LeftBound25691.bound (LeftBound25691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority25693.bound LeftBound25691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25693.bound, LeftBound25691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority25693.actual selector witness) * (LeftBound25691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25697

namespace LeftBound25705
def owner : Owner := ⟨.program ⟨214⟩, ⟨16233⟩⟩
def transferEvent : Nat := 25705
def frameStart : Nat := 25632
def rule : BoundRule := .sum [.predecessor 0 25703 .coefficient, .predecessor 1 25704 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25703 .coefficient)
      LeftAuthority25701.bound (LeftAuthority25701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25704 .coefficient)
      LeftBound25697.bound (LeftBound25697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25697.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25701.bound, LeftBound25697.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25701.bound, LeftBound25697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25701.actual selector witness, LeftBound25697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25705

namespace LeftBound25709
def owner : Owner := ⟨.program ⟨214⟩, ⟨28340⟩⟩
def transferEvent : Nat := 25709
def frameStart : Nat := 25632
def rule : BoundRule := .product (.predecessor 0 25707 .coefficient) (.predecessor 1 25708 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25707 .coefficient)
      LeftBound25705.bound (LeftBound25705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25708 .coefficient)
      LeftAuthority25682.bound (LeftAuthority25682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound25705.bound LeftAuthority25682.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25705.bound, LeftAuthority25682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound25705.actual selector witness) * (LeftAuthority25682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25709

namespace LeftBound25720
def owner : Owner := ⟨.program ⟨214⟩, ⟨18390⟩⟩
def transferEvent : Nat := 25720
def frameStart : Nat := 25632
def rule : BoundRule := .product (.predecessor 0 25718 .coefficient) (.predecessor 1 25719 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25718 .coefficient)
      LeftAuthority25693.bound (LeftAuthority25693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25694RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25693.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25693.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25719 .coefficient)
      LeftAuthority25716.bound (LeftAuthority25716.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25717RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25716.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25716.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority25693.bound LeftAuthority25716.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25693.bound, LeftAuthority25716.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority25693.actual selector witness) * (LeftAuthority25716.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25720

namespace LeftBound25728
def owner : Owner := ⟨.program ⟨214⟩, ⟨18391⟩⟩
def transferEvent : Nat := 25728
def frameStart : Nat := 25632
def rule : BoundRule := .sum [.predecessor 0 25726 .coefficient, .predecessor 1 25727 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25726 .coefficient)
      LeftAuthority25724.bound (LeftAuthority25724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority25724.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority25724.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25727 .coefficient)
      LeftBound25720.bound (LeftBound25720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25722RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority25724.bound, LeftBound25720.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority25724.bound, LeftBound25720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority25724.actual selector witness, LeftBound25720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25728

namespace LeftBound25732
def owner : Owner := ⟨.program ⟨214⟩, ⟨28344⟩⟩
def transferEvent : Nat := 25732
def frameStart : Nat := 25632
def rule : BoundRule := .sum [.predecessor 0 25730 .coefficient, .predecessor 1 25731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25730 .coefficient)
      LeftBound25728.bound (LeftBound25728.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25728.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25728.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25731 .coefficient)
      LeftBound25709.bound (LeftBound25709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25728.bound, LeftBound25709.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25728.bound, LeftBound25709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25728.actual selector witness, LeftBound25709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25732

namespace LeftBound25745
def owner : Owner := ⟨.program ⟨214⟩, ⟨28342⟩⟩
def transferEvent : Nat := 25745
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25743 .coefficient, .predecessor 1 25744 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25743 .coefficient)
      LeftBound25574.bound (LeftBound25574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25574.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25744 .coefficient)
      LeftBound25557.bound (LeftBound25557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25557.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25557.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25574.bound, LeftBound25557.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25574.bound, LeftBound25557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25574.actual selector witness, LeftBound25557.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25745

namespace LeftBound25748
def owner : Owner := ⟨.program ⟨214⟩, ⟨28342⟩⟩
def transferEvent : Nat := 25748
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 25742 .summary, .result 25564 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25742 .summary)
      LeftBound25576.bound (LeftBound25576.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21703⟩⟩) (rawTerms := some (Proof.Events100.exact25742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25576.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 25564 .summary)
      LeftBound25559.bound (LeftBound25559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28341⟩⟩) (rawTerms := some (Proof.Events099.exact25564RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound25559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25576.bound, LeftBound25559.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25576.bound, LeftBound25559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25576.actual selector witness, LeftBound25559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25748

namespace LeftBound25772
def owner : Owner := ⟨.program ⟨214⟩, ⟨11566⟩⟩
def transferEvent : Nat := 25772
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 25770 .coefficient) (.predecessor 1 25771 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25770 .coefficient)
      LeftAuthority1048.bound (LeftAuthority1048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1048.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25771 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1048.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1048.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1048.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound25772

namespace LeftBound25777
def owner : Owner := ⟨.program ⟨214⟩, ⟨7350⟩⟩
def transferEvent : Nat := 25777
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25775 .coefficient) (.predecessor 1 25776 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25775 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25776 .coefficient)
      LeftBound10980.bound (LeftBound10980.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10981RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10980.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10980.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound10980.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound10980.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound10980.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25777

namespace LeftBound25782
def owner : Owner := ⟨.program ⟨214⟩, ⟨11567⟩⟩
def transferEvent : Nat := 25782
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25780 .coefficient, .predecessor 1 25781 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25780 .coefficient)
      LeftBound25777.bound (LeftBound25777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25777.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25781 .coefficient)
      LeftBound25772.bound (LeftBound25772.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25772.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25772.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25777.bound, LeftBound25772.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25777.bound, LeftBound25772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25777.actual selector witness, LeftBound25772.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25782

namespace LeftBound25786
def owner : Owner := ⟨.program ⟨214⟩, ⟨11568⟩⟩
def transferEvent : Nat := 25786
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 25784 .coefficient, .predecessor 1 25785 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25784 .coefficient)
      LeftBound25782.bound (LeftBound25782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25782.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25782.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25785 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound25782.bound, LeftBound10972.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25782.bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound25782.actual selector witness, LeftBound10972.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound25786

namespace LeftBound25787
def owner : Owner := ⟨.program ⟨214⟩, ⟨11568⟩⟩
def transferEvent : Nat := 25787
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨94⟩⟩]⟩ [⟨.result 10973 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10973 .coefficient)
      LeftBound10972.bound (LeftBound10972.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨94⟩⟩) (rawTerms := some (Proof.Events042.exact10973RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10972.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10972.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound10972.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10972.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound10972.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25787

namespace LeftBound25792
def owner : Owner := ⟨.program ⟨214⟩, ⟨14454⟩⟩
def transferEvent : Nat := 25792
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 25790 .coefficient) (.predecessor 1 25791 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 25790 .coefficient)
      LeftBound25786.bound (LeftBound25786.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events100.exact25789RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25786.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25786.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 25791 .coefficient)
      LeftAuthority1051.bound (LeftAuthority1051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound25786.bound LeftAuthority1051.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25786.bound, LeftAuthority1051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound25786.actual selector witness) * (LeftAuthority1051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound25792

namespace LeftBound25793
def owner : Owner := ⟨.program ⟨214⟩, ⟨14454⟩⟩
def transferEvent : Nat := 25793
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨14451⟩⟩], []⟩ [⟨.result 1052 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 1052 .coefficient)
      LeftAuthority1051.bound (LeftAuthority1051.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨14451⟩⟩) (rawTerms := some (Proof.Events004.exact1052RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1051.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1051.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority1051.bound []
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority1051.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound25793

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
