import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard702

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101827
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def transferEvent : Nat := 101827
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24898⟩⟩]⟩ [⟨.result 101759 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101759 .coefficient)
      LeftAuthority101758.bound (LeftAuthority101758.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24898⟩⟩) (rawTerms := some (Proof.Events397.exact101759RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101758.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101758.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101758.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101758.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101827

namespace LeftBound101828
def owner : Owner := ⟨.program ⟨214⟩, ⟨24899⟩⟩
def transferEvent : Nat := 101828
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101823 .summary) (.transfer 101827) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101823 .summary)
      LeftBound101822.bound (LeftBound101822.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10463⟩⟩) (rawTerms := some (Proof.Events397.exact101823RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101827)
      LeftBound101827.bound (LeftBound101827.actual selector witness) := by
  exact .transfer (LeftBound101827.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101822.bound LeftBound101827.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101822.bound, LeftBound101827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101822.actual selector witness) * (LeftBound101827.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101828

namespace LeftBound101839
def owner : Owner := ⟨.program ⟨214⟩, ⟨19015⟩⟩
def transferEvent : Nat := 101839
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 101837 .coefficient) (.value (.predecessor 1 101838 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101837 .coefficient)
      LeftAuthority101835.bound (LeftAuthority101835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101838 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101835.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101835.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101835.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101839

namespace LeftBound101843
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def transferEvent : Nat := 101843
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101841 .coefficient) (.predecessor 1 101842 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101841 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101842 .coefficient)
      LeftBound101839.bound (LeftBound101839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events397.exact101840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound101839.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound101839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound101839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101843

namespace LeftBound101844
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def transferEvent : Nat := 101844
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19013⟩⟩]⟩ [⟨.result 101836 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101836 .coefficient)
      LeftAuthority101835.bound (LeftAuthority101835.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19013⟩⟩) (rawTerms := some (Proof.Events397.exact101836RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101835.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101835.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101835.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101844

namespace LeftBound101845
def owner : Owner := ⟨.program ⟨214⟩, ⟨19016⟩⟩
def transferEvent : Nat := 101845
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 101844) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101844)
      LeftBound101844.bound (LeftBound101844.actual selector witness) := by
  exact .transfer (LeftBound101844.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound101844.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound101844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound101844.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101845

namespace LeftBound101900
def owner : Owner := ⟨.program ⟨214⟩, ⟨10457⟩⟩
def transferEvent : Nat := 101900
def frameStart : Nat := 101883
def rule : BoundRule := .product (.predecessor 0 101898 .coefficient) (.predecessor 1 101899 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101898 .coefficient)
      LeftAuthority101896.bound (LeftAuthority101896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101896.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101899 .coefficient)
      LeftAuthority101893.bound (LeftAuthority101893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101893.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101896.bound LeftAuthority101893.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101896.bound, LeftAuthority101893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101896.actual selector witness) * (LeftAuthority101893.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101900

namespace LeftBound101904
def owner : Owner := ⟨.program ⟨214⟩, ⟨10458⟩⟩
def transferEvent : Nat := 101904
def frameStart : Nat := 101883
def rule : BoundRule := .identity (.predecessor 0 101903 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101903 .coefficient)
      LeftBound101900.bound (LeftBound101900.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101900.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101900.derived selector witness)

def rawBound : CoeffClass := LeftBound101900.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101900.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101900.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101904

namespace LeftBound101921
def owner : Owner := ⟨.program ⟨214⟩, ⟨10568⟩⟩
def transferEvent : Nat := 101921
def frameStart : Nat := 101883
def rule : BoundRule := .sum [.predecessor 0 101919 .coefficient, .predecessor 1 101920 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101919 .coefficient)
      LeftBound101904.bound (LeftBound101904.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101920 .coefficient)
      LeftAuthority101917.bound (LeftAuthority101917.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101904.bound, LeftAuthority101917.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101904.bound, LeftAuthority101917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101904.actual selector witness, LeftAuthority101917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101921

namespace LeftBound101924
def owner : Owner := ⟨.program ⟨214⟩, ⟨10569⟩⟩
def transferEvent : Nat := 101924
def frameStart : Nat := 101883
def rule : BoundRule := .identity (.predecessor 0 101923 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101923 .coefficient)
      LeftBound101921.bound (LeftBound101921.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101921.derived selector witness)

def rawBound : CoeffClass := LeftBound101921.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101921.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101921.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101924

namespace LeftBound101930
def owner : Owner := ⟨.program ⟨214⟩, ⟨10570⟩⟩
def transferEvent : Nat := 101930
def frameStart : Nat := 101883
def rule : BoundRule := .product (.predecessor 0 101928 .coefficient) (.predecessor 1 101929 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101928 .coefficient)
      LeftAuthority101926.bound (LeftAuthority101926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101929 .coefficient)
      LeftBound101924.bound (LeftBound101924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101924.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority101926.bound LeftBound101924.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101926.bound, LeftBound101924.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority101926.actual selector witness) * (LeftBound101924.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101930

namespace LeftBound101946
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 101946
def frameStart : Nat := 101883
def rule : BoundRule := .scale (.predecessor 0 101944 .coefficient) (.value (.predecessor 1 101945 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101944 .coefficient)
      LeftAuthority101942.bound (LeftAuthority101942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101945 .coefficient)
      LeftAuthority101933.bound (LeftAuthority101933.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101933.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101942.bound LeftAuthority101933.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101942.bound, LeftAuthority101933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101942.actual selector witness) * (LeftAuthority101933.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101946

namespace LeftBound101949
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 101949
def frameStart : Nat := 101883
def rule : BoundRule := .identity (.predecessor 0 101948 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101948 .coefficient)
      LeftAuthority101936.bound (LeftAuthority101936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101936.derived selector witness)

def rawBound : CoeffClass := LeftAuthority101936.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority101936.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101949

namespace LeftBound101953
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def transferEvent : Nat := 101953
def frameStart : Nat := 101883
def rule : BoundRule := .product (.predecessor 0 101951 .coefficient) (.predecessor 1 101952 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101951 .coefficient)
      LeftBound101949.bound (LeftBound101949.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101949.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101949.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101952 .coefficient)
      LeftBound101946.bound (LeftBound101946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101946.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101949.bound LeftBound101946.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101949.bound, LeftBound101946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101949.actual selector witness) * (LeftBound101946.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101953

namespace LeftBound101958
def owner : Owner := ⟨.program ⟨214⟩, ⟨10571⟩⟩
def transferEvent : Nat := 101958
def frameStart : Nat := 101883
def rule : BoundRule := .sum [.predecessor 0 101956 .coefficient, .predecessor 1 101957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101956 .coefficient)
      LeftBound101953.bound (LeftBound101953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101957 .coefficient)
      LeftBound101930.bound (LeftBound101930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101953.bound, LeftBound101930.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101953.bound, LeftBound101930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101953.actual selector witness, LeftBound101930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101958

namespace LeftBound101962
def owner : Owner := ⟨.program ⟨214⟩, ⟨24901⟩⟩
def transferEvent : Nat := 101962
def frameStart : Nat := 101883
def rule : BoundRule := .product (.predecessor 0 101960 .coefficient) (.predecessor 1 101961 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101960 .coefficient)
      LeftBound101958.bound (LeftBound101958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101961 .coefficient)
      LeftAuthority101915.bound (LeftAuthority101915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events398.exact101916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101958.bound LeftAuthority101915.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101958.bound, LeftAuthority101915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101958.actual selector witness) * (LeftAuthority101915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101962

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
