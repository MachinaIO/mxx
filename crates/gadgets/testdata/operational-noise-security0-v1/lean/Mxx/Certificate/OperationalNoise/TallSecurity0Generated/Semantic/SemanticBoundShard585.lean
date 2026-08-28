import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard076
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard538
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard584

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85654
def owner : Owner := ⟨.program ⟨214⟩, ⟨27654⟩⟩
def transferEvent : Nat := 85654
def frameStart : Nat := 85554
def rule : BoundRule := .sum [.predecessor 0 85652 .coefficient, .predecessor 1 85653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85652 .coefficient)
      LeftBound85650.bound (LeftBound85650.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85651RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85650.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85650.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85653 .coefficient)
      LeftBound85631.bound (LeftBound85631.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85631.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85631.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85650.bound, LeftBound85631.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85650.bound, LeftBound85631.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85650.actual selector witness, LeftBound85631.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85654

namespace LeftBound85667
def owner : Owner := ⟨.program ⟨214⟩, ⟨27652⟩⟩
def transferEvent : Nat := 85667
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85665 .coefficient, .predecessor 1 85666 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85665 .coefficient)
      LeftBound85496.bound (LeftBound85496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85496.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85666 .coefficient)
      LeftBound85479.bound (LeftBound85479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events333.exact85486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85496.bound, LeftBound85479.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85496.bound, LeftBound85479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85496.actual selector witness, LeftBound85479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85667

namespace LeftBound85670
def owner : Owner := ⟨.program ⟨214⟩, ⟨27652⟩⟩
def transferEvent : Nat := 85670
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 85664 .summary, .result 85486 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85664 .summary)
      LeftBound85498.bound (LeftBound85498.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21259⟩⟩) (rawTerms := some (Proof.Events334.exact85664RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85486 .summary)
      LeftBound85481.bound (LeftBound85481.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27651⟩⟩) (rawTerms := some (Proof.Events333.exact85486RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85481.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85498.bound, LeftBound85481.bound]
def bound : CoeffClass := .finite ⟨1292046061494565744640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85498.bound, LeftBound85481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85498.actual selector witness, LeftBound85481.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85670

namespace LeftBound85694
def owner : Owner := ⟨.program ⟨214⟩, ⟨11302⟩⟩
def transferEvent : Nat := 85694
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 85692 .coefficient) (.predecessor 1 85693 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85692 .coefficient)
      LeftAuthority4103.bound (LeftAuthority4103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85693 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4103.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4103.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4103.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85694

namespace LeftBound85699
def owner : Owner := ⟨.program ⟨214⟩, ⟨7233⟩⟩
def transferEvent : Nat := 85699
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85697 .coefficient) (.predecessor 1 85698 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85697 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85698 .coefficient)
      LeftBound12483.bound (LeftBound12483.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12483.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12483.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound12483.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound12483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound12483.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85699

namespace LeftBound85704
def owner : Owner := ⟨.program ⟨214⟩, ⟨11303⟩⟩
def transferEvent : Nat := 85704
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85702 .coefficient, .predecessor 1 85703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85702 .coefficient)
      LeftBound85699.bound (LeftBound85699.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85699.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85699.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85703 .coefficient)
      LeftBound85694.bound (LeftBound85694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85694.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85699.bound, LeftBound85694.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85699.bound, LeftBound85694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85699.actual selector witness, LeftBound85694.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85704

namespace LeftBound85708
def owner : Owner := ⟨.program ⟨214⟩, ⟨11304⟩⟩
def transferEvent : Nat := 85708
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85706 .coefficient, .predecessor 1 85707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85706 .coefficient)
      LeftBound85704.bound (LeftBound85704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85707 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85704.bound, LeftBound12475.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85704.bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85704.actual selector witness, LeftBound12475.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85708

namespace LeftBound85709
def owner : Owner := ⟨.program ⟨214⟩, ⟨11304⟩⟩
def transferEvent : Nat := 85709
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨91⟩⟩]⟩ [⟨.result 12476 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12476 .coefficient)
      LeftBound12475.bound (LeftBound12475.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨91⟩⟩) (rawTerms := some (Proof.Events048.exact12476RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12475.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12475.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12475.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12475.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12475.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85709

namespace LeftBound85714
def owner : Owner := ⟨.program ⟨214⟩, ⟨13776⟩⟩
def transferEvent : Nat := 85714
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85712 .coefficient) (.predecessor 1 85713 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85712 .coefficient)
      LeftBound85708.bound (LeftBound85708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85708.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85713 .coefficient)
      LeftAuthority4106.bound (LeftAuthority4106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound85708.bound LeftAuthority4106.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85708.bound, LeftAuthority4106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound85708.actual selector witness) * (LeftAuthority4106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85714

namespace LeftBound85715
def owner : Owner := ⟨.program ⟨214⟩, ⟨13776⟩⟩
def transferEvent : Nat := 85715
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13773⟩⟩], []⟩ [⟨.result 4107 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4107 .coefficient)
      LeftAuthority4106.bound (LeftAuthority4106.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13773⟩⟩) (rawTerms := some (Proof.Events016.exact4107RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4106.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4106.bound []
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4106.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85715

namespace LeftBound85716
def owner : Owner := ⟨.program ⟨214⟩, ⟨13776⟩⟩
def transferEvent : Nat := 85716
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 85711 .summary) (.transfer 85715) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85711 .summary)
      LeftBound85709.bound (LeftBound85709.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11304⟩⟩) (rawTerms := some (Proof.Events334.exact85711RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85709.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85715)
      LeftBound85715.bound (LeftBound85715.actual selector witness) := by
  exact .transfer (LeftBound85715.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound85709.bound LeftBound85715.bound
def bound : CoeffClass := .finite ⟨9984, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85709.bound, LeftBound85715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound85709.actual selector witness) * (LeftBound85715.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85716

namespace LeftBound85722
def owner : Owner := ⟨.program ⟨214⟩, ⟨13777⟩⟩
def transferEvent : Nat := 85722
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 85720 .coefficient) (.predecessor 1 85721 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85720 .coefficient)
      LeftAuthority4106.bound (LeftAuthority4106.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85721 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4106.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4106.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4106.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85722

namespace LeftBound85727
def owner : Owner := ⟨.program ⟨214⟩, ⟨7250⟩⟩
def transferEvent : Nat := 85727
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85725 .coefficient) (.predecessor 1 85726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85725 .coefficient)
      LeftBound79789.bound (LeftBound79789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events311.exact79790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85726 .coefficient)
      LeftBound12524.bound (LeftBound12524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound79789.bound LeftBound12524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound79789.bound, LeftBound12524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound79789.actual selector witness) * (LeftBound12524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85727

namespace LeftBound85732
def owner : Owner := ⟨.program ⟨214⟩, ⟨13778⟩⟩
def transferEvent : Nat := 85732
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85730 .coefficient, .predecessor 1 85731 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85730 .coefficient)
      LeftBound85727.bound (LeftBound85727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85729RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85727.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85731 .coefficient)
      LeftBound85722.bound (LeftBound85722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85727.bound, LeftBound85722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85727.bound, LeftBound85722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85727.actual selector witness, LeftBound85722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85732

namespace LeftBound85736
def owner : Owner := ⟨.program ⟨214⟩, ⟨13779⟩⟩
def transferEvent : Nat := 85736
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 85734 .coefficient, .predecessor 1 85735 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85734 .coefficient)
      LeftBound85732.bound (LeftBound85732.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events334.exact85733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85732.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85732.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85735 .coefficient)
      LeftBound12516.bound (LeftBound12516.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12517RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12516.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12516.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85732.bound, LeftBound12516.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85732.bound, LeftBound12516.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85732.actual selector witness, LeftBound12516.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85736

namespace LeftBound85737
def owner : Owner := ⟨.program ⟨214⟩, ⟨13779⟩⟩
def transferEvent : Nat := 85737
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
end LeftBound85737

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
