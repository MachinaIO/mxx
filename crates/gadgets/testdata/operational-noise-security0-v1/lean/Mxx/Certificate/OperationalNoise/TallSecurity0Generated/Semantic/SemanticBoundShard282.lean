import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard281

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound41907
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def transferEvent : Nat := 41907
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41905 .coefficient) (.predecessor 1 41906 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41905 .coefficient)
      LeftBound41901.bound (LeftBound41901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41904RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41901.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41901.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41906 .coefficient)
      LeftAuthority41839.bound (LeftAuthority41839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41901.bound LeftAuthority41839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41901.bound, LeftAuthority41839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41901.actual selector witness) * (LeftAuthority41839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41907

namespace LeftBound41908
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def transferEvent : Nat := 41908
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25922⟩⟩]⟩ [⟨.result 41840 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41840 .coefficient)
      LeftAuthority41839.bound (LeftAuthority41839.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25922⟩⟩) (rawTerms := some (Proof.Events163.exact41840RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41839.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41839.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41839.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41908

namespace LeftBound41909
def owner : Owner := ⟨.program ⟨214⟩, ⟨25923⟩⟩
def transferEvent : Nat := 41909
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41904 .summary) (.transfer 41908) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41904 .summary)
      LeftBound41903.bound (LeftBound41903.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13799⟩⟩) (rawTerms := some (Proof.Events163.exact41904RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41903.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41908)
      LeftBound41908.bound (LeftBound41908.actual selector witness) := by
  exact .transfer (LeftBound41908.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41903.bound LeftBound41908.bound
def bound : CoeffClass := .finite ⟨350231094886400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41903.bound, LeftBound41908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41903.actual selector witness) * (LeftBound41908.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41909

namespace LeftBound41920
def owner : Owner := ⟨.program ⟨214⟩, ⟨19394⟩⟩
def transferEvent : Nat := 41920
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 41918 .coefficient) (.value (.predecessor 1 41919 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41918 .coefficient)
      LeftAuthority41916.bound (LeftAuthority41916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41917RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41916.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41919 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority41916.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41916.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41916.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound41920

namespace LeftBound41924
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def transferEvent : Nat := 41924
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 41922 .coefficient) (.predecessor 1 41923 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 41922 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 41923 .coefficient)
      LeftBound41920.bound (LeftBound41920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events163.exact41921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41920.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41920.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound41920.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound41920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound41920.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41924

namespace LeftBound41925
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def transferEvent : Nat := 41925
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19392⟩⟩]⟩ [⟨.result 41917 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41917 .coefficient)
      LeftAuthority41916.bound (LeftAuthority41916.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19392⟩⟩) (rawTerms := some (Proof.Events163.exact41917RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41916.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41916.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority41916.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority41916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority41916.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound41925

namespace LeftBound41926
def owner : Owner := ⟨.program ⟨214⟩, ⟨19395⟩⟩
def transferEvent : Nat := 41926
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 41925) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 41925)
      LeftBound41925.bound (LeftBound41925.actual selector witness) := by
  exact .transfer (LeftBound41925.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound41925.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound41925.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound41925.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound41926

namespace LeftBound42005
def owner : Owner := ⟨.program ⟨214⟩, ⟨13792⟩⟩
def transferEvent : Nat := 42005
def frameStart : Nat := 41976
def rule : BoundRule := .product (.predecessor 0 42003 .coefficient) (.predecessor 1 42004 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42003 .coefficient)
      LeftAuthority42001.bound (LeftAuthority42001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42001.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42001.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42004 .coefficient)
      LeftAuthority41998.bound (LeftAuthority41998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact41999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority41998.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority41998.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority42001.bound LeftAuthority41998.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42001.bound, LeftAuthority41998.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority42001.actual selector witness) * (LeftAuthority41998.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42005

namespace LeftBound42009
def owner : Owner := ⟨.program ⟨214⟩, ⟨13793⟩⟩
def transferEvent : Nat := 42009
def frameStart : Nat := 41976
def rule : BoundRule := .identity (.predecessor 0 42008 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42008 .coefficient)
      LeftBound42005.bound (LeftBound42005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42005.derived selector witness)

def rawBound : CoeffClass := LeftBound42005.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42005.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42009

namespace LeftBound42026
def owner : Owner := ⟨.program ⟨214⟩, ⟨13888⟩⟩
def transferEvent : Nat := 42026
def frameStart : Nat := 41976
def rule : BoundRule := .sum [.predecessor 0 42024 .coefficient, .predecessor 1 42025 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42024 .coefficient)
      LeftBound42009.bound (LeftBound42009.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42025 .coefficient)
      LeftAuthority42022.bound (LeftAuthority42022.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42022.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42009.bound, LeftAuthority42022.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42009.bound, LeftAuthority42022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42009.actual selector witness, LeftAuthority42022.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42026

namespace LeftBound42029
def owner : Owner := ⟨.program ⟨214⟩, ⟨13889⟩⟩
def transferEvent : Nat := 42029
def frameStart : Nat := 41976
def rule : BoundRule := .identity (.predecessor 0 42028 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42028 .coefficient)
      LeftBound42026.bound (LeftBound42026.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound42026.derived selector witness)

def rawBound : CoeffClass := LeftBound42026.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound42026.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42029

namespace LeftBound42035
def owner : Owner := ⟨.program ⟨214⟩, ⟨13890⟩⟩
def transferEvent : Nat := 42035
def frameStart : Nat := 41976
def rule : BoundRule := .product (.predecessor 0 42033 .coefficient) (.predecessor 1 42034 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42033 .coefficient)
      LeftAuthority42031.bound (LeftAuthority42031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42031.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42034 .coefficient)
      LeftBound42029.bound (LeftBound42029.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42030RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42029.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42029.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority42031.bound LeftBound42029.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42031.bound, LeftBound42029.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority42031.actual selector witness) * (LeftBound42029.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42035

namespace LeftBound42051
def owner : Owner := ⟨.program ⟨214⟩, ⟨7847⟩⟩
def transferEvent : Nat := 42051
def frameStart : Nat := 41976
def rule : BoundRule := .scale (.predecessor 0 42049 .coefficient) (.value (.predecessor 1 42050 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42049 .coefficient)
      LeftAuthority42047.bound (LeftAuthority42047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42050 .coefficient)
      LeftAuthority42038.bound (LeftAuthority42038.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority42038.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority42047.bound LeftAuthority42038.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42047.bound, LeftAuthority42038.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority42047.actual selector witness) * (LeftAuthority42038.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound42051

namespace LeftBound42054
def owner : Owner := ⟨.program ⟨214⟩, ⟨6794⟩⟩
def transferEvent : Nat := 42054
def frameStart : Nat := 41976
def rule : BoundRule := .identity (.predecessor 0 42053 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42053 .coefficient)
      LeftAuthority42041.bound (LeftAuthority42041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42042RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority42041.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority42041.derived selector witness)

def rawBound : CoeffClass := LeftAuthority42041.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority42041.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority42041.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound42054

namespace LeftBound42058
def owner : Owner := ⟨.program ⟨214⟩, ⟨7848⟩⟩
def transferEvent : Nat := 42058
def frameStart : Nat := 41976
def rule : BoundRule := .product (.predecessor 0 42056 .coefficient) (.predecessor 1 42057 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42056 .coefficient)
      LeftBound42054.bound (LeftBound42054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42057 .coefficient)
      LeftBound42051.bound (LeftBound42051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42054.bound LeftBound42051.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42054.bound, LeftBound42051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42054.actual selector witness) * (LeftBound42051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound42058

namespace LeftBound42063
def owner : Owner := ⟨.program ⟨214⟩, ⟨13891⟩⟩
def transferEvent : Nat := 42063
def frameStart : Nat := 41976
def rule : BoundRule := .sum [.predecessor 0 42061 .coefficient, .predecessor 1 42062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 42061 .coefficient)
      LeftBound42058.bound (LeftBound42058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 42062 .coefficient)
      LeftBound42035.bound (LeftBound42035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events164.exact42037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound42035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound42035.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound42058.bound, LeftBound42035.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42058.bound, LeftBound42035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound42058.actual selector witness, LeftBound42035.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound42063

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
