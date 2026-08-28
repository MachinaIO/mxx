import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard115

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18812
def owner : Owner := ⟨.program ⟨214⟩, ⟨21635⟩⟩
def transferEvent : Nat := 18812
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18810 .coefficient) (.predecessor 1 18811 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18810 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18811 .coefficient)
      LeftBound18808.bound (LeftBound18808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18808.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18808.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound18808.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound18808.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound18808.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18812

namespace LeftBound18813
def owner : Owner := ⟨.program ⟨214⟩, ⟨21635⟩⟩
def transferEvent : Nat := 18813
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21632⟩⟩]⟩ [⟨.result 18805 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18805 .coefficient)
      LeftAuthority18804.bound (LeftAuthority18804.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21632⟩⟩) (rawTerms := some (Proof.Events073.exact18805RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18804.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18804.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18804.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18804.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18813

namespace LeftBound18814
def owner : Owner := ⟨.program ⟨214⟩, ⟨21635⟩⟩
def transferEvent : Nat := 18814
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 18813) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18813)
      LeftBound18813.bound (LeftBound18813.actual selector witness) := by
  exact .transfer (LeftBound18813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound18813.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound18813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound18813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18814

namespace LeftBound18909
def owner : Owner := ⟨.program ⟨214⟩, ⟨16195⟩⟩
def transferEvent : Nat := 18909
def frameStart : Nat := 18870
def rule : BoundRule := .identity (.predecessor 0 18908 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18908 .coefficient)
      LeftAuthority18906.bound (LeftAuthority18906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18906.derived selector witness)

def rawBound : CoeffClass := LeftAuthority18906.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority18906.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18909

namespace LeftBound18926
def owner : Owner := ⟨.program ⟨214⟩, ⟨16234⟩⟩
def transferEvent : Nat := 18926
def frameStart : Nat := 18870
def rule : BoundRule := .sum [.predecessor 0 18924 .coefficient, .predecessor 1 18925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18924 .coefficient)
      LeftBound18909.bound (LeftBound18909.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18909.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18925 .coefficient)
      LeftAuthority18922.bound (LeftAuthority18922.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority18922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18909.bound, LeftAuthority18922.bound]
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18909.bound, LeftAuthority18922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18909.actual selector witness, LeftAuthority18922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18926

namespace LeftBound18929
def owner : Owner := ⟨.program ⟨214⟩, ⟨16235⟩⟩
def transferEvent : Nat := 18929
def frameStart : Nat := 18870
def rule : BoundRule := .identity (.predecessor 0 18928 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18928 .coefficient)
      LeftBound18926.bound (LeftBound18926.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18926.derived selector witness)

def rawBound : CoeffClass := LeftBound18926.bound
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound18926.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18929

namespace LeftBound18935
def owner : Owner := ⟨.program ⟨214⟩, ⟨16236⟩⟩
def transferEvent : Nat := 18935
def frameStart : Nat := 18870
def rule : BoundRule := .product (.predecessor 0 18933 .coefficient) (.predecessor 1 18934 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18933 .coefficient)
      LeftAuthority18931.bound (LeftAuthority18931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18934 .coefficient)
      LeftBound18929.bound (LeftBound18929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18929.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority18931.bound LeftBound18929.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18931.bound, LeftBound18929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority18931.actual selector witness) * (LeftBound18929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18935

namespace LeftBound18943
def owner : Owner := ⟨.program ⟨214⟩, ⟨16237⟩⟩
def transferEvent : Nat := 18943
def frameStart : Nat := 18870
def rule : BoundRule := .sum [.predecessor 0 18941 .coefficient, .predecessor 1 18942 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18941 .coefficient)
      LeftAuthority18939.bound (LeftAuthority18939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18942 .coefficient)
      LeftBound18935.bound (LeftBound18935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18935.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18939.bound, LeftBound18935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18939.bound, LeftBound18935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18939.actual selector witness, LeftBound18935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18943

namespace LeftBound18947
def owner : Owner := ⟨.program ⟨214⟩, ⟨28346⟩⟩
def transferEvent : Nat := 18947
def frameStart : Nat := 18870
def rule : BoundRule := .product (.predecessor 0 18945 .coefficient) (.predecessor 1 18946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18945 .coefficient)
      LeftBound18943.bound (LeftBound18943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18946 .coefficient)
      LeftAuthority18920.bound (LeftAuthority18920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18943.bound LeftAuthority18920.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18943.bound, LeftAuthority18920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18943.actual selector witness) * (LeftAuthority18920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18947

namespace LeftBound18958
def owner : Owner := ⟨.program ⟨214⟩, ⟨17680⟩⟩
def transferEvent : Nat := 18958
def frameStart : Nat := 18870
def rule : BoundRule := .product (.predecessor 0 18956 .coefficient) (.predecessor 1 18957 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18956 .coefficient)
      LeftAuthority18931.bound (LeftAuthority18931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18931.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18957 .coefficient)
      LeftAuthority18954.bound (LeftAuthority18954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18954.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18954.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18931.bound LeftAuthority18954.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18931.bound, LeftAuthority18954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority18931.actual selector witness) * (LeftAuthority18954.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18958

namespace LeftBound18966
def owner : Owner := ⟨.program ⟨214⟩, ⟨17681⟩⟩
def transferEvent : Nat := 18966
def frameStart : Nat := 18870
def rule : BoundRule := .sum [.predecessor 0 18964 .coefficient, .predecessor 1 18965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18964 .coefficient)
      LeftAuthority18962.bound (LeftAuthority18962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18962.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18965 .coefficient)
      LeftBound18958.bound (LeftBound18958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18958.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18962.bound, LeftBound18958.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18962.bound, LeftBound18958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18962.actual selector witness, LeftBound18958.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18966

namespace LeftBound18970
def owner : Owner := ⟨.program ⟨214⟩, ⟨28351⟩⟩
def transferEvent : Nat := 18970
def frameStart : Nat := 18870
def rule : BoundRule := .sum [.predecessor 0 18968 .coefficient, .predecessor 1 18969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18968 .coefficient)
      LeftBound18966.bound (LeftBound18966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18969 .coefficient)
      LeftBound18947.bound (LeftBound18947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18947.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18966.bound, LeftBound18947.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18966.bound, LeftBound18947.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18966.actual selector witness, LeftBound18947.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18970

namespace LeftBound18983
def owner : Owner := ⟨.program ⟨214⟩, ⟨28348⟩⟩
def transferEvent : Nat := 18983
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 18981 .coefficient, .predecessor 1 18982 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18981 .coefficient)
      LeftBound18812.bound (LeftBound18812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18812.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18812.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18982 .coefficient)
      LeftBound18795.bound (LeftBound18795.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events073.exact18802RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18795.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18795.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18812.bound, LeftBound18795.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18812.bound, LeftBound18795.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18812.actual selector witness, LeftBound18795.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18983

namespace LeftBound18986
def owner : Owner := ⟨.program ⟨214⟩, ⟨28348⟩⟩
def transferEvent : Nat := 18986
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 18980 .summary, .result 18802 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18980 .summary)
      LeftBound18814.bound (LeftBound18814.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21635⟩⟩) (rawTerms := some (Proof.Events074.exact18980RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18814.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18802 .summary)
      LeftBound18797.bound (LeftBound18797.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28347⟩⟩) (rawTerms := some (Proof.Events073.exact18802RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18797.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18814.bound, LeftBound18797.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18814.bound, LeftBound18797.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18814.actual selector witness, LeftBound18797.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18986

namespace LeftBound18990
def owner : Owner := ⟨.program ⟨214⟩, ⟨28349⟩⟩
def transferEvent : Nat := 18990
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18988 .coefficient) (.predecessor 1 18989 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18988 .coefficient)
      LeftBound18983.bound (LeftBound18983.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact18987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18983.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18983.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18989 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18983.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18983.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18983.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18990

namespace LeftBound18991
def owner : Owner := ⟨.program ⟨214⟩, ⟨28349⟩⟩
def transferEvent : Nat := 18991
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18991

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
