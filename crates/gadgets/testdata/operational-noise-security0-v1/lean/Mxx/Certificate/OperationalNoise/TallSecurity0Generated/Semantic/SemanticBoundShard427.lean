import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard426

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63751
def owner : Owner := ⟨.program ⟨214⟩, ⟨20903⟩⟩
def transferEvent : Nat := 63751
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20900⟩⟩]⟩ [⟨.result 63743 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63743 .coefficient)
      LeftAuthority63742.bound (LeftAuthority63742.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20900⟩⟩) (rawTerms := some (Proof.Events248.exact63743RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63742.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63742.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63742.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63742.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63742.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63751

namespace LeftBound63752
def owner : Owner := ⟨.program ⟨214⟩, ⟨20903⟩⟩
def transferEvent : Nat := 63752
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 63751) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63751)
      LeftBound63751.bound (LeftBound63751.actual selector witness) := by
  exact .transfer (LeftBound63751.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound63751.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound63751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound63751.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63752

namespace LeftBound63847
def owner : Owner := ⟨.program ⟨214⟩, ⟨15588⟩⟩
def transferEvent : Nat := 63847
def frameStart : Nat := 63808
def rule : BoundRule := .identity (.predecessor 0 63846 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63846 .coefficient)
      LeftAuthority63844.bound (LeftAuthority63844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63844.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63844.derived selector witness)

def rawBound : CoeffClass := LeftAuthority63844.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority63844.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63847

namespace LeftBound63864
def owner : Owner := ⟨.program ⟨214⟩, ⟨15662⟩⟩
def transferEvent : Nat := 63864
def frameStart : Nat := 63808
def rule : BoundRule := .sum [.predecessor 0 63862 .coefficient, .predecessor 1 63863 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63862 .coefficient)
      LeftBound63847.bound (LeftBound63847.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63847.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63863 .coefficient)
      LeftAuthority63860.bound (LeftAuthority63860.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63860.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63847.bound, LeftAuthority63860.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63847.bound, LeftAuthority63860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63847.actual selector witness, LeftAuthority63860.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63864

namespace LeftBound63867
def owner : Owner := ⟨.program ⟨214⟩, ⟨15663⟩⟩
def transferEvent : Nat := 63867
def frameStart : Nat := 63808
def rule : BoundRule := .identity (.predecessor 0 63866 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63866 .coefficient)
      LeftBound63864.bound (LeftBound63864.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63864.derived selector witness)

def rawBound : CoeffClass := LeftBound63864.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63864.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound63864.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63867

namespace LeftBound63873
def owner : Owner := ⟨.program ⟨214⟩, ⟨15664⟩⟩
def transferEvent : Nat := 63873
def frameStart : Nat := 63808
def rule : BoundRule := .product (.predecessor 0 63871 .coefficient) (.predecessor 1 63872 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63871 .coefficient)
      LeftAuthority63869.bound (LeftAuthority63869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63872 .coefficient)
      LeftBound63867.bound (LeftBound63867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63867.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority63869.bound LeftBound63867.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63869.bound, LeftBound63867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority63869.actual selector witness) * (LeftBound63867.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63873

namespace LeftBound63881
def owner : Owner := ⟨.program ⟨214⟩, ⟨15665⟩⟩
def transferEvent : Nat := 63881
def frameStart : Nat := 63808
def rule : BoundRule := .sum [.predecessor 0 63879 .coefficient, .predecessor 1 63880 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63879 .coefficient)
      LeftAuthority63877.bound (LeftAuthority63877.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63877.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63877.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63880 .coefficient)
      LeftBound63873.bound (LeftBound63873.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63873.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63873.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63877.bound, LeftBound63873.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63877.bound, LeftBound63873.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63877.actual selector witness, LeftBound63873.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63881

namespace LeftBound63885
def owner : Owner := ⟨.program ⟨214⟩, ⟨27222⟩⟩
def transferEvent : Nat := 63885
def frameStart : Nat := 63808
def rule : BoundRule := .product (.predecessor 0 63883 .coefficient) (.predecessor 1 63884 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63883 .coefficient)
      LeftBound63881.bound (LeftBound63881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63881.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63881.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63884 .coefficient)
      LeftAuthority63858.bound (LeftAuthority63858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63858.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63881.bound LeftAuthority63858.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63881.bound, LeftAuthority63858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63881.actual selector witness) * (LeftAuthority63858.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63885

namespace LeftBound63896
def owner : Owner := ⟨.program ⟨214⟩, ⟨17828⟩⟩
def transferEvent : Nat := 63896
def frameStart : Nat := 63808
def rule : BoundRule := .product (.predecessor 0 63894 .coefficient) (.predecessor 1 63895 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63894 .coefficient)
      LeftAuthority63869.bound (LeftAuthority63869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63869.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63895 .coefficient)
      LeftAuthority63892.bound (LeftAuthority63892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63869.bound LeftAuthority63892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63869.bound, LeftAuthority63892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority63869.actual selector witness) * (LeftAuthority63892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63896

namespace LeftBound63904
def owner : Owner := ⟨.program ⟨214⟩, ⟨17829⟩⟩
def transferEvent : Nat := 63904
def frameStart : Nat := 63808
def rule : BoundRule := .sum [.predecessor 0 63902 .coefficient, .predecessor 1 63903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63902 .coefficient)
      LeftAuthority63900.bound (LeftAuthority63900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63900.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63900.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63903 .coefficient)
      LeftBound63896.bound (LeftBound63896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63900.bound, LeftBound63896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63900.bound, LeftBound63896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63900.actual selector witness, LeftBound63896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63904

namespace LeftBound63908
def owner : Owner := ⟨.program ⟨214⟩, ⟨27227⟩⟩
def transferEvent : Nat := 63908
def frameStart : Nat := 63808
def rule : BoundRule := .sum [.predecessor 0 63906 .coefficient, .predecessor 1 63907 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63906 .coefficient)
      LeftBound63904.bound (LeftBound63904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63907 .coefficient)
      LeftBound63885.bound (LeftBound63885.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63890RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63885.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63885.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63904.bound, LeftBound63885.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63904.bound, LeftBound63885.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63904.actual selector witness, LeftBound63885.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63908

namespace LeftBound63921
def owner : Owner := ⟨.program ⟨214⟩, ⟨27224⟩⟩
def transferEvent : Nat := 63921
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63919 .coefficient, .predecessor 1 63920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63919 .coefficient)
      LeftBound63750.bound (LeftBound63750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63920 .coefficient)
      LeftBound63733.bound (LeftBound63733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events248.exact63740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63750.bound, LeftBound63733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63750.bound, LeftBound63733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63750.actual selector witness, LeftBound63733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63921

namespace LeftBound63924
def owner : Owner := ⟨.program ⟨214⟩, ⟨27224⟩⟩
def transferEvent : Nat := 63924
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63918 .summary, .result 63740 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63918 .summary)
      LeftBound63752.bound (LeftBound63752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20903⟩⟩) (rawTerms := some (Proof.Events249.exact63918RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63740 .summary)
      LeftBound63735.bound (LeftBound63735.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27223⟩⟩) (rawTerms := some (Proof.Events248.exact63740RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63752.bound, LeftBound63735.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63752.bound, LeftBound63735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63752.actual selector witness, LeftBound63735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63924

namespace LeftBound63928
def owner : Owner := ⟨.program ⟨214⟩, ⟨27225⟩⟩
def transferEvent : Nat := 63928
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63926 .coefficient) (.predecessor 1 63927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63926 .coefficient)
      LeftBound63921.bound (LeftBound63921.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events249.exact63925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63921.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63927 .coefficient)
      LeftBound5778.bound (LeftBound5778.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5779RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5778.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5778.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63921.bound LeftBound5778.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63921.bound, LeftBound5778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63921.actual selector witness) * (LeftBound5778.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63928

namespace LeftBound63929
def owner : Owner := ⟨.program ⟨214⟩, ⟨27225⟩⟩
def transferEvent : Nat := 63929
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6649⟩⟩]⟩ [⟨.result 5775 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5775 .coefficient)
      LeftAuthority5774.bound (LeftAuthority5774.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6649⟩⟩) (rawTerms := some (Proof.Events022.exact5775RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5774.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5774.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5774.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5774.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5774.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63929

namespace LeftBound63930
def owner : Owner := ⟨.program ⟨214⟩, ⟨27225⟩⟩
def transferEvent : Nat := 63930
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 63925 .summary) (.transfer 63929) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63925 .summary)
      LeftBound63924.bound (LeftBound63924.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27224⟩⟩) (rawTerms := some (Proof.Events249.exact63925RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63929)
      LeftBound63929.bound (LeftBound63929.actual selector witness) := by
  exact .transfer (LeftBound63929.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63924.bound LeftBound63929.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63924.bound, LeftBound63929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63924.actual selector witness) * (LeftBound63929.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63930

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
