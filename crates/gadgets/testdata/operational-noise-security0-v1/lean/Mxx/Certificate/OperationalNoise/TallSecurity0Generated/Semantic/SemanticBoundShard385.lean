import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard384

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56748
def owner : Owner := ⟨.program ⟨214⟩, ⟨21118⟩⟩
def transferEvent : Nat := 56748
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 56746 .coefficient) (.value (.predecessor 1 56747 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56746 .coefficient)
      LeftAuthority56744.bound (LeftAuthority56744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56747 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56744.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56744.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56744.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56748

namespace LeftBound56752
def owner : Owner := ⟨.program ⟨214⟩, ⟨21119⟩⟩
def transferEvent : Nat := 56752
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56750 .coefficient) (.predecessor 1 56751 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56750 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56751 .coefficient)
      LeftBound56748.bound (LeftBound56748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound56748.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound56748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound56748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56752

namespace LeftBound56753
def owner : Owner := ⟨.program ⟨214⟩, ⟨21119⟩⟩
def transferEvent : Nat := 56753
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21116⟩⟩]⟩ [⟨.result 56745 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56745 .coefficient)
      LeftAuthority56744.bound (LeftAuthority56744.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21116⟩⟩) (rawTerms := some (Proof.Events221.exact56745RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56744.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56744.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56744.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56753

namespace LeftBound56754
def owner : Owner := ⟨.program ⟨214⟩, ⟨21119⟩⟩
def transferEvent : Nat := 56754
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 56753) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56753)
      LeftBound56753.bound (LeftBound56753.actual selector witness) := by
  exact .transfer (LeftBound56753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound56753.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound56753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound56753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56754

namespace LeftBound56849
def owner : Owner := ⟨.program ⟨214⟩, ⟨15707⟩⟩
def transferEvent : Nat := 56849
def frameStart : Nat := 56810
def rule : BoundRule := .identity (.predecessor 0 56848 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56848 .coefficient)
      LeftAuthority56846.bound (LeftAuthority56846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56846.derived selector witness)

def rawBound : CoeffClass := LeftAuthority56846.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority56846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56849

namespace LeftBound56866
def owner : Owner := ⟨.program ⟨214⟩, ⟨15781⟩⟩
def transferEvent : Nat := 56866
def frameStart : Nat := 56810
def rule : BoundRule := .sum [.predecessor 0 56864 .coefficient, .predecessor 1 56865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56864 .coefficient)
      LeftBound56849.bound (LeftBound56849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56865 .coefficient)
      LeftAuthority56862.bound (LeftAuthority56862.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56849.bound, LeftAuthority56862.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56849.bound, LeftAuthority56862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56849.actual selector witness, LeftAuthority56862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56866

namespace LeftBound56869
def owner : Owner := ⟨.program ⟨214⟩, ⟨15782⟩⟩
def transferEvent : Nat := 56869
def frameStart : Nat := 56810
def rule : BoundRule := .identity (.predecessor 0 56868 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56868 .coefficient)
      LeftBound56866.bound (LeftBound56866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56866.derived selector witness)

def rawBound : CoeffClass := LeftBound56866.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56866.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56869

namespace LeftBound56875
def owner : Owner := ⟨.program ⟨214⟩, ⟨15783⟩⟩
def transferEvent : Nat := 56875
def frameStart : Nat := 56810
def rule : BoundRule := .product (.predecessor 0 56873 .coefficient) (.predecessor 1 56874 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56873 .coefficient)
      LeftAuthority56871.bound (LeftAuthority56871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56874 .coefficient)
      LeftBound56869.bound (LeftBound56869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority56871.bound LeftBound56869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56871.bound, LeftBound56869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority56871.actual selector witness) * (LeftBound56869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56875

namespace LeftBound56883
def owner : Owner := ⟨.program ⟨214⟩, ⟨15784⟩⟩
def transferEvent : Nat := 56883
def frameStart : Nat := 56810
def rule : BoundRule := .sum [.predecessor 0 56881 .coefficient, .predecessor 1 56882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56881 .coefficient)
      LeftAuthority56879.bound (LeftAuthority56879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56882 .coefficient)
      LeftBound56875.bound (LeftBound56875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56879.bound, LeftBound56875.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56879.bound, LeftBound56875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority56879.actual selector witness, LeftBound56875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56883

namespace LeftBound56887
def owner : Owner := ⟨.program ⟨214⟩, ⟨27446⟩⟩
def transferEvent : Nat := 56887
def frameStart : Nat := 56810
def rule : BoundRule := .product (.predecessor 0 56885 .coefficient) (.predecessor 1 56886 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56885 .coefficient)
      LeftBound56883.bound (LeftBound56883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56886 .coefficient)
      LeftAuthority56860.bound (LeftAuthority56860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56883.bound LeftAuthority56860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56883.bound, LeftAuthority56860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56883.actual selector witness) * (LeftAuthority56860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56887

namespace LeftBound56898
def owner : Owner := ⟨.program ⟨214⟩, ⟨15752⟩⟩
def transferEvent : Nat := 56898
def frameStart : Nat := 56810
def rule : BoundRule := .product (.predecessor 0 56896 .coefficient) (.predecessor 1 56897 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56896 .coefficient)
      LeftAuthority56871.bound (LeftAuthority56871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56897 .coefficient)
      LeftAuthority56894.bound (LeftAuthority56894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56871.bound LeftAuthority56894.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56871.bound, LeftAuthority56894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority56871.actual selector witness) * (LeftAuthority56894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56898

namespace LeftBound56906
def owner : Owner := ⟨.program ⟨214⟩, ⟨15753⟩⟩
def transferEvent : Nat := 56906
def frameStart : Nat := 56810
def rule : BoundRule := .sum [.predecessor 0 56904 .coefficient, .predecessor 1 56905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56904 .coefficient)
      LeftAuthority56902.bound (LeftAuthority56902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56905 .coefficient)
      LeftBound56898.bound (LeftBound56898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56902.bound, LeftBound56898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56902.bound, LeftBound56898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority56902.actual selector witness, LeftBound56898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56906

namespace LeftBound56910
def owner : Owner := ⟨.program ⟨214⟩, ⟨27450⟩⟩
def transferEvent : Nat := 56910
def frameStart : Nat := 56810
def rule : BoundRule := .sum [.predecessor 0 56908 .coefficient, .predecessor 1 56909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56908 .coefficient)
      LeftBound56906.bound (LeftBound56906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56909 .coefficient)
      LeftBound56887.bound (LeftBound56887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56906.bound, LeftBound56887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56906.bound, LeftBound56887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56906.actual selector witness, LeftBound56887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56910

namespace LeftBound56923
def owner : Owner := ⟨.program ⟨214⟩, ⟨27448⟩⟩
def transferEvent : Nat := 56923
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 56921 .coefficient, .predecessor 1 56922 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56921 .coefficient)
      LeftBound56752.bound (LeftBound56752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events222.exact56920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56922 .coefficient)
      LeftBound56735.bound (LeftBound56735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56752.bound, LeftBound56735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56752.bound, LeftBound56735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56752.actual selector witness, LeftBound56735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56923

namespace LeftBound56926
def owner : Owner := ⟨.program ⟨214⟩, ⟨27448⟩⟩
def transferEvent : Nat := 56926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 56920 .summary, .result 56742 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56920 .summary)
      LeftBound56754.bound (LeftBound56754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21119⟩⟩) (rawTerms := some (Proof.Events222.exact56920RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56742 .summary)
      LeftBound56737.bound (LeftBound56737.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27447⟩⟩) (rawTerms := some (Proof.Events221.exact56742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound56737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56754.bound, LeftBound56737.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56754.bound, LeftBound56737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56754.actual selector witness, LeftBound56737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56926

namespace LeftBound56950
def owner : Owner := ⟨.program ⟨214⟩, ⟨11222⟩⟩
def transferEvent : Nat := 56950
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 56948 .coefficient) (.predecessor 1 56949 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56948 .coefficient)
      LeftAuthority2636.bound (LeftAuthority2636.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events010.exact2637RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2636.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2636.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56949 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2636.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2636.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2636.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56950

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
