import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard096
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard132
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard135
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard197

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound29650
def owner : Owner := ⟨.program ⟨214⟩, ⟨10510⟩⟩
def transferEvent : Nat := 29650
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29645 .summary) (.transfer 29649) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29645 .summary)
      LeftBound29643.bound (LeftBound29643.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10509⟩⟩) (rawTerms := some (Proof.Events115.exact29645RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29643.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29649)
      LeftBound29649.bound (LeftBound29649.actual selector witness) := by
  exact .transfer (LeftBound29649.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound29643.bound LeftBound29649.bound
def bound : CoeffClass := .finite ⟨1664, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29643.bound, LeftBound29649.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound29643.actual selector witness) * (LeftBound29649.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29650

namespace LeftBound29656
def owner : Owner := ⟨.program ⟨214⟩, ⟨9416⟩⟩
def transferEvent : Nat := 29656
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 29654 .coefficient) (.predecessor 1 29655 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29654 .coefficient)
      LeftAuthority1235.bound (LeftAuthority1235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events004.exact1236RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority1235.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority1235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29655 .coefficient)
      LeftBound21418.bound (LeftBound21418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21418.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21418.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority1235.bound LeftBound21418.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority1235.bound, LeftBound21418.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority1235.actual selector witness) * (LeftBound21418.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29656

namespace LeftBound29661
def owner : Owner := ⟨.program ⟨214⟩, ⟨7341⟩⟩
def transferEvent : Nat := 29661
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29659 .coefficient) (.predecessor 1 29660 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29659 .coefficient)
      LeftBound21289.bound (LeftBound21289.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events083.exact21290RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21289.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21289.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29660 .coefficient)
      LeftBound15029.bound (LeftBound15029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound21289.bound LeftBound15029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21289.bound, LeftBound15029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound21289.actual selector witness) * (LeftBound15029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29661

namespace LeftBound29666
def owner : Owner := ⟨.program ⟨214⟩, ⟨9417⟩⟩
def transferEvent : Nat := 29666
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29664 .coefficient, .predecessor 1 29665 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29664 .coefficient)
      LeftBound29661.bound (LeftBound29661.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29663RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29661.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29665 .coefficient)
      LeftBound29656.bound (LeftBound29656.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29656.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29656.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29661.bound, LeftBound29656.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29661.bound, LeftBound29656.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29661.actual selector witness, LeftBound29656.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29666

namespace LeftBound29670
def owner : Owner := ⟨.program ⟨214⟩, ⟨9418⟩⟩
def transferEvent : Nat := 29670
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29668 .coefficient, .predecessor 1 29669 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29668 .coefficient)
      LeftBound29666.bound (LeftBound29666.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29667RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29666.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29666.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29669 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29666.bound, LeftBound15021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29666.bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29666.actual selector witness, LeftBound15021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29670

namespace LeftBound29671
def owner : Owner := ⟨.program ⟨214⟩, ⟨9418⟩⟩
def transferEvent : Nat := 29671
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨85⟩⟩]⟩ [⟨.result 15022 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15022 .coefficient)
      LeftBound15021.bound (LeftBound15021.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨85⟩⟩) (rawTerms := some (Proof.Events058.exact15022RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15021.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound15021.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound15021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound15021.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29671

namespace LeftBound29676
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def transferEvent : Nat := 29676
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29674 .coefficient) (.predecessor 1 29675 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29674 .coefficient)
      LeftBound29670.bound (LeftBound29670.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29673RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29670.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29670.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29675 .coefficient)
      LeftBound15018.bound (LeftBound15018.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events058.exact15019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15018.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15018.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29670.bound LeftBound15018.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29670.bound, LeftBound15018.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29670.actual selector witness) * (LeftBound15018.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29676

namespace LeftBound29677
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def transferEvent : Nat := 29677
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩ [⟨.result 15015 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 15015 .coefficient)
      LeftAuthority15014.bound (LeftAuthority15014.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7831⟩⟩) (rawTerms := some (Proof.Events058.exact15015RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15014.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15014.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15014.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority15014.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29677

namespace LeftBound29678
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def transferEvent : Nat := 29678
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29673 .summary) (.transfer 29677) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29673 .summary)
      LeftBound29671.bound (LeftBound29671.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9418⟩⟩) (rawTerms := some (Proof.Events115.exact29673RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29677)
      LeftBound29677.bound (LeftBound29677.actual selector witness) := by
  exact .transfer (LeftBound29677.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29671.bound LeftBound29677.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29671.bound, LeftBound29677.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29671.actual selector witness) * (LeftBound29677.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29678

namespace LeftBound29686
def owner : Owner := ⟨.program ⟨214⟩, ⟨10511⟩⟩
def transferEvent : Nat := 29686
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 29684 .coefficient, .predecessor 1 29685 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29684 .coefficient)
      LeftBound29676.bound (LeftBound29676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29676.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29676.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29685 .coefficient)
      LeftBound29648.bound (LeftBound29648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29648.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29648.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29676.bound, LeftBound29648.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29676.bound, LeftBound29648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29676.actual selector witness, LeftBound29648.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29686

namespace LeftBound29688
def owner : Owner := ⟨.program ⟨214⟩, ⟨10511⟩⟩
def transferEvent : Nat := 29688
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 29683 .summary, .result 29653 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29683 .summary)
      LeftBound29678.bound (LeftBound29678.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9419⟩⟩) (rawTerms := some (Proof.Events115.exact29683RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29678.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29653 .summary)
      LeftBound29650.bound (LeftBound29650.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10510⟩⟩) (rawTerms := some (Proof.Events115.exact29653RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound29678.bound, LeftBound29650.bound]
def bound : CoeffClass := .finite ⟨95422080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29678.bound, LeftBound29650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound29678.actual selector witness, LeftBound29650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound29688

namespace LeftBound29692
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def transferEvent : Nat := 29692
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29690 .coefficient) (.predecessor 1 29691 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29690 .coefficient)
      LeftBound29686.bound (LeftBound29686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29691 .coefficient)
      LeftAuthority29624.bound (LeftAuthority29624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events115.exact29625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29624.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29686.bound LeftAuthority29624.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29686.bound, LeftAuthority29624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29686.actual selector witness) * (LeftAuthority29624.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29692

namespace LeftBound29693
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def transferEvent : Nat := 29693
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24926⟩⟩]⟩ [⟨.result 29625 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29625 .coefficient)
      LeftAuthority29624.bound (LeftAuthority29624.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24926⟩⟩) (rawTerms := some (Proof.Events115.exact29625RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29624.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29624.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority29624.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29624.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29624.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound29693

namespace LeftBound29694
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def transferEvent : Nat := 29694
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29689 .summary) (.transfer 29693) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29689 .summary)
      LeftBound29688.bound (LeftBound29688.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10511⟩⟩) (rawTerms := some (Proof.Events115.exact29689RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 29693)
      LeftBound29693.bound (LeftBound29693.actual selector witness) := by
  exact .transfer (LeftBound29693.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29688.bound LeftBound29693.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29688.bound, LeftBound29693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29688.actual selector witness) * (LeftBound29693.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29694

namespace LeftBound29705
def owner : Owner := ⟨.program ⟨214⟩, ⟨19038⟩⟩
def transferEvent : Nat := 29705
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 29703 .coefficient) (.value (.predecessor 1 29704 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29703 .coefficient)
      LeftAuthority29701.bound (LeftAuthority29701.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29702RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority29701.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority29701.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29704 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority29701.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority29701.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority29701.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound29705

namespace LeftBound29709
def owner : Owner := ⟨.program ⟨214⟩, ⟨19039⟩⟩
def transferEvent : Nat := 29709
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 29707 .coefficient) (.predecessor 1 29708 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 29707 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 29708 .coefficient)
      LeftBound29705.bound (LeftBound29705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29706RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29705.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound29705.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound29705.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound29705.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound29709

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
