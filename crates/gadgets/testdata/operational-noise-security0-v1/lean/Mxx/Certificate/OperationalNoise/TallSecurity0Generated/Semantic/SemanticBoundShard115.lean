import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard063
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard114

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18717
def owner : Owner := ⟨.program ⟨214⟩, ⟨16354⟩⟩
def transferEvent : Nat := 18717
def frameStart : Nat := 18658
def rule : BoundRule := .identity (.predecessor 0 18716 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18716 .coefficient)
      LeftBound18714.bound (LeftBound18714.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18714.derived selector witness)

def rawBound : CoeffClass := LeftBound18714.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound18714.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18717

namespace LeftBound18723
def owner : Owner := ⟨.program ⟨214⟩, ⟨16355⟩⟩
def transferEvent : Nat := 18723
def frameStart : Nat := 18658
def rule : BoundRule := .product (.predecessor 0 18721 .coefficient) (.predecessor 1 18722 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18721 .coefficient)
      LeftAuthority18719.bound (LeftAuthority18719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18722 .coefficient)
      LeftBound18717.bound (LeftBound18717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18717.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18717.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority18719.bound LeftBound18717.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18719.bound, LeftBound18717.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority18719.actual selector witness) * (LeftBound18717.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18723

namespace LeftBound18731
def owner : Owner := ⟨.program ⟨214⟩, ⟨16356⟩⟩
def transferEvent : Nat := 18731
def frameStart : Nat := 18658
def rule : BoundRule := .sum [.predecessor 0 18729 .coefficient, .predecessor 1 18730 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18729 .coefficient)
      LeftAuthority18727.bound (LeftAuthority18727.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18728RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18727.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18727.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18730 .coefficient)
      LeftBound18723.bound (LeftBound18723.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18723.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18723.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18727.bound, LeftBound18723.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18727.bound, LeftBound18723.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18727.actual selector witness, LeftBound18723.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18731

namespace LeftBound18735
def owner : Owner := ⟨.program ⟨214⟩, ⟨28563⟩⟩
def transferEvent : Nat := 18735
def frameStart : Nat := 18658
def rule : BoundRule := .product (.predecessor 0 18733 .coefficient) (.predecessor 1 18734 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18733 .coefficient)
      LeftBound18731.bound (LeftBound18731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18731.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18731.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18734 .coefficient)
      LeftAuthority18708.bound (LeftAuthority18708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18708.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18708.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18731.bound LeftAuthority18708.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18731.bound, LeftAuthority18708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18731.actual selector witness) * (LeftAuthority18708.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18735

namespace LeftBound18746
def owner : Owner := ⟨.program ⟨214⟩, ⟨17624⟩⟩
def transferEvent : Nat := 18746
def frameStart : Nat := 18658
def rule : BoundRule := .product (.predecessor 0 18744 .coefficient) (.predecessor 1 18745 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18744 .coefficient)
      LeftAuthority18719.bound (LeftAuthority18719.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18720RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18719.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18719.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18745 .coefficient)
      LeftAuthority18742.bound (LeftAuthority18742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18743RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18742.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18719.bound LeftAuthority18742.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18719.bound, LeftAuthority18742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority18719.actual selector witness) * (LeftAuthority18742.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18746

namespace LeftBound18754
def owner : Owner := ⟨.program ⟨214⟩, ⟨17625⟩⟩
def transferEvent : Nat := 18754
def frameStart : Nat := 18658
def rule : BoundRule := .sum [.predecessor 0 18752 .coefficient, .predecessor 1 18753 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18752 .coefficient)
      LeftAuthority18750.bound (LeftAuthority18750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18753 .coefficient)
      LeftBound18746.bound (LeftBound18746.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18746.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18746.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18750.bound, LeftBound18746.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18750.bound, LeftBound18746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18750.actual selector witness, LeftBound18746.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18754

namespace LeftBound18758
def owner : Owner := ⟨.program ⟨214⟩, ⟨28568⟩⟩
def transferEvent : Nat := 18758
def frameStart : Nat := 18658
def rule : BoundRule := .sum [.predecessor 0 18756 .coefficient, .predecessor 1 18757 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18756 .coefficient)
      LeftBound18754.bound (LeftBound18754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18757 .coefficient)
      LeftBound18735.bound (LeftBound18735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18754.bound, LeftBound18735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18754.bound, LeftBound18735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18754.actual selector witness, LeftBound18735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18758

namespace LeftBound18771
def owner : Owner := ⟨.program ⟨214⟩, ⟨28565⟩⟩
def transferEvent : Nat := 18771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 18769 .coefficient, .predecessor 1 18770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18769 .coefficient)
      LeftBound18600.bound (LeftBound18600.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18600.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18600.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18770 .coefficient)
      LeftBound18583.bound (LeftBound18583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18583.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18600.bound, LeftBound18583.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18600.bound, LeftBound18583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18600.actual selector witness, LeftBound18583.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18771

namespace LeftBound18774
def owner : Owner := ⟨.program ⟨214⟩, ⟨28565⟩⟩
def transferEvent : Nat := 18774
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 18768 .summary, .result 18590 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18768 .summary)
      LeftBound18602.bound (LeftBound18602.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21779⟩⟩) (rawTerms := some (Proof.Events073.exact18768RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18590 .summary)
      LeftBound18585.bound (LeftBound18585.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28564⟩⟩) (rawTerms := some (Proof.Events072.exact18590RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18585.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18602.bound, LeftBound18585.bound]
def bound : CoeffClass := .finite ⟨1292202948609709846528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18602.bound, LeftBound18585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18602.actual selector witness, LeftBound18585.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18774

namespace LeftBound18778
def owner : Owner := ⟨.program ⟨214⟩, ⟨28566⟩⟩
def transferEvent : Nat := 18778
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18776 .coefficient) (.predecessor 1 18777 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18776 .coefficient)
      LeftBound18771.bound (LeftBound18771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18775RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18777 .coefficient)
      LeftBound5658.bound (LeftBound5658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5659RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5658.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18771.bound LeftBound5658.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18771.bound, LeftBound5658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18771.actual selector witness) * (LeftBound5658.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18778

namespace LeftBound18779
def owner : Owner := ⟨.program ⟨214⟩, ⟨28566⟩⟩
def transferEvent : Nat := 18779
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6677⟩⟩]⟩ [⟨.result 5655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5655 .coefficient)
      LeftAuthority5654.bound (LeftAuthority5654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6677⟩⟩) (rawTerms := some (Proof.Events022.exact5655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18779

namespace LeftBound18780
def owner : Owner := ⟨.program ⟨214⟩, ⟨28566⟩⟩
def transferEvent : Nat := 18780
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 18775 .summary) (.transfer 18779) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18775 .summary)
      LeftBound18774.bound (LeftBound18774.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28565⟩⟩) (rawTerms := some (Proof.Events073.exact18775RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18774.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18779)
      LeftBound18779.bound (LeftBound18779.actual selector witness) := by
  exact .transfer (LeftBound18779.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18774.bound LeftBound18779.bound
def bound : CoeffClass := .finite ⟨4742405496644812892115304448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18774.bound, LeftBound18779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18774.actual selector witness) * (LeftBound18779.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18780

namespace LeftBound18795
def owner : Owner := ⟨.program ⟨214⟩, ⟨28347⟩⟩
def transferEvent : Nat := 18795
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18793 .coefficient) (.predecessor 1 18794 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18793 .coefficient)
      LeftBound10751.bound (LeftBound10751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events042.exact10755RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound10751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound10751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18794 .coefficient)
      LeftAuthority18791.bound (LeftAuthority18791.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18792RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18791.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10751.bound LeftAuthority18791.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10751.bound, LeftAuthority18791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10751.actual selector witness) * (LeftAuthority18791.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18795

namespace LeftBound18796
def owner : Owner := ⟨.program ⟨214⟩, ⟨28347⟩⟩
def transferEvent : Nat := 18796
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28345⟩⟩]⟩ [⟨.result 18792 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18792 .coefficient)
      LeftAuthority18791.bound (LeftAuthority18791.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28345⟩⟩) (rawTerms := some (Proof.Events073.exact18792RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18791.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18791.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18791.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18791.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18791.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18796

namespace LeftBound18797
def owner : Owner := ⟨.program ⟨214⟩, ⟨28347⟩⟩
def transferEvent : Nat := 18797
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 10755 .summary) (.transfer 18796) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 10755 .summary)
      LeftBound10754.bound (LeftBound10754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26242⟩⟩) (rawTerms := some (Proof.Events042.exact10755RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound10754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18796)
      LeftBound18796.bound (LeftBound18796.actual selector witness) := by
  exact .transfer (LeftBound18796.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound10754.bound LeftBound18796.bound
def bound : CoeffClass := .finite ⟨1292180534353385750528, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound10754.bound, LeftBound18796.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound10754.actual selector witness) * (LeftBound18796.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18797

namespace LeftBound18808
def owner : Owner := ⟨.program ⟨214⟩, ⟨21634⟩⟩
def transferEvent : Nat := 18808
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 18806 .coefficient) (.value (.predecessor 1 18807 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18806 .coefficient)
      LeftAuthority18804.bound (LeftAuthority18804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18804.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18807 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority18804.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18804.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18804.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound18808

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
