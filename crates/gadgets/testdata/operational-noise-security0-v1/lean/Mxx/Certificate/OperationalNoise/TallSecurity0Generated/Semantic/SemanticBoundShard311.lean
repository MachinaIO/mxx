import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard310

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46789
def owner : Owner := ⟨.program ⟨214⟩, ⟨22490⟩⟩
def transferEvent : Nat := 46789
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 46787 .coefficient) (.value (.predecessor 1 46788 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46787 .coefficient)
      LeftAuthority46785.bound (LeftAuthority46785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46788 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority46785.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46785.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46785.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound46789

namespace LeftBound46793
def owner : Owner := ⟨.program ⟨214⟩, ⟨22491⟩⟩
def transferEvent : Nat := 46793
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46791 .coefficient) (.predecessor 1 46792 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46791 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46792 .coefficient)
      LeftBound46789.bound (LeftBound46789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46789.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound46789.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound46789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound46789.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46793

namespace LeftBound46794
def owner : Owner := ⟨.program ⟨214⟩, ⟨22491⟩⟩
def transferEvent : Nat := 46794
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22488⟩⟩]⟩ [⟨.result 46786 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46786 .coefficient)
      LeftAuthority46785.bound (LeftAuthority46785.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22488⟩⟩) (rawTerms := some (Proof.Events182.exact46786RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46785.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46785.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46785.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46785.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46794

namespace LeftBound46795
def owner : Owner := ⟨.program ⟨214⟩, ⟨22491⟩⟩
def transferEvent : Nat := 46795
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 46794) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46794)
      LeftBound46794.bound (LeftBound46794.actual selector witness) := by
  exact .transfer (LeftBound46794.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound46794.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound46794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound46794.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46795

namespace LeftBound46890
def owner : Owner := ⟨.program ⟨214⟩, ⟨16761⟩⟩
def transferEvent : Nat := 46890
def frameStart : Nat := 46851
def rule : BoundRule := .identity (.predecessor 0 46889 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46889 .coefficient)
      LeftAuthority46887.bound (LeftAuthority46887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46887.derived selector witness)

def rawBound : CoeffClass := LeftAuthority46887.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority46887.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46890

namespace LeftBound46907
def owner : Owner := ⟨.program ⟨214⟩, ⟨16835⟩⟩
def transferEvent : Nat := 46907
def frameStart : Nat := 46851
def rule : BoundRule := .sum [.predecessor 0 46905 .coefficient, .predecessor 1 46906 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46905 .coefficient)
      LeftBound46890.bound (LeftBound46890.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46890.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46906 .coefficient)
      LeftAuthority46903.bound (LeftAuthority46903.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority46903.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46890.bound, LeftAuthority46903.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46890.bound, LeftAuthority46903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46890.actual selector witness, LeftAuthority46903.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46907

namespace LeftBound46910
def owner : Owner := ⟨.program ⟨214⟩, ⟨16836⟩⟩
def transferEvent : Nat := 46910
def frameStart : Nat := 46851
def rule : BoundRule := .identity (.predecessor 0 46909 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46909 .coefficient)
      LeftBound46907.bound (LeftBound46907.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46907.derived selector witness)

def rawBound : CoeffClass := LeftBound46907.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46907.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound46907.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46910

namespace LeftBound46916
def owner : Owner := ⟨.program ⟨214⟩, ⟨16837⟩⟩
def transferEvent : Nat := 46916
def frameStart : Nat := 46851
def rule : BoundRule := .product (.predecessor 0 46914 .coefficient) (.predecessor 1 46915 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46914 .coefficient)
      LeftAuthority46912.bound (LeftAuthority46912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46915 .coefficient)
      LeftBound46910.bound (LeftBound46910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46910.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority46912.bound LeftBound46910.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46912.bound, LeftBound46910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority46912.actual selector witness) * (LeftBound46910.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46916

namespace LeftBound46924
def owner : Owner := ⟨.program ⟨214⟩, ⟨16838⟩⟩
def transferEvent : Nat := 46924
def frameStart : Nat := 46851
def rule : BoundRule := .sum [.predecessor 0 46922 .coefficient, .predecessor 1 46923 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46922 .coefficient)
      LeftAuthority46920.bound (LeftAuthority46920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46923 .coefficient)
      LeftBound46916.bound (LeftBound46916.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46916.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46916.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46920.bound, LeftBound46916.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46920.bound, LeftBound46916.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46920.actual selector witness, LeftBound46916.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46924

namespace LeftBound46928
def owner : Owner := ⟨.program ⟨214⟩, ⟨29622⟩⟩
def transferEvent : Nat := 46928
def frameStart : Nat := 46851
def rule : BoundRule := .product (.predecessor 0 46926 .coefficient) (.predecessor 1 46927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46926 .coefficient)
      LeftBound46924.bound (LeftBound46924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46924.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46927 .coefficient)
      LeftAuthority46901.bound (LeftAuthority46901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46901.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46924.bound LeftAuthority46901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46924.bound, LeftAuthority46901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46924.actual selector witness) * (LeftAuthority46901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46928

namespace LeftBound46939
def owner : Owner := ⟨.program ⟨214⟩, ⟨17504⟩⟩
def transferEvent : Nat := 46939
def frameStart : Nat := 46851
def rule : BoundRule := .product (.predecessor 0 46937 .coefficient) (.predecessor 1 46938 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46937 .coefficient)
      LeftAuthority46912.bound (LeftAuthority46912.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46912.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46912.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46938 .coefficient)
      LeftAuthority46935.bound (LeftAuthority46935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46935.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority46912.bound LeftAuthority46935.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46912.bound, LeftAuthority46935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority46912.actual selector witness) * (LeftAuthority46935.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46939

namespace LeftBound46947
def owner : Owner := ⟨.program ⟨214⟩, ⟨17505⟩⟩
def transferEvent : Nat := 46947
def frameStart : Nat := 46851
def rule : BoundRule := .sum [.predecessor 0 46945 .coefficient, .predecessor 1 46946 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46945 .coefficient)
      LeftAuthority46943.bound (LeftAuthority46943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46946 .coefficient)
      LeftBound46939.bound (LeftBound46939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46939.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46939.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46943.bound, LeftBound46939.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46943.bound, LeftBound46939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46943.actual selector witness, LeftBound46939.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46947

namespace LeftBound46951
def owner : Owner := ⟨.program ⟨214⟩, ⟨29627⟩⟩
def transferEvent : Nat := 46951
def frameStart : Nat := 46851
def rule : BoundRule := .sum [.predecessor 0 46949 .coefficient, .predecessor 1 46950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46949 .coefficient)
      LeftBound46947.bound (LeftBound46947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46950 .coefficient)
      LeftBound46928.bound (LeftBound46928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46947.bound, LeftBound46928.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46947.bound, LeftBound46928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46947.actual selector witness, LeftBound46928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46951

namespace LeftBound46964
def owner : Owner := ⟨.program ⟨214⟩, ⟨29624⟩⟩
def transferEvent : Nat := 46964
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 46962 .coefficient, .predecessor 1 46963 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46962 .coefficient)
      LeftBound46793.bound (LeftBound46793.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46793.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46793.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46963 .coefficient)
      LeftBound46776.bound (LeftBound46776.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46776.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46776.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46793.bound, LeftBound46776.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46793.bound, LeftBound46776.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46793.actual selector witness, LeftBound46776.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46964

namespace LeftBound46967
def owner : Owner := ⟨.program ⟨214⟩, ⟨29624⟩⟩
def transferEvent : Nat := 46967
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 46961 .summary, .result 46783 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46961 .summary)
      LeftBound46795.bound (LeftBound46795.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22491⟩⟩) (rawTerms := some (Proof.Events183.exact46961RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46795.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46783 .summary)
      LeftBound46778.bound (LeftBound46778.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29623⟩⟩) (rawTerms := some (Proof.Events182.exact46783RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46778.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46795.bound, LeftBound46778.bound]
def bound : CoeffClass := .finite ⟨1292449485504936292352, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46795.bound, LeftBound46778.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46795.actual selector witness, LeftBound46778.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46967

namespace LeftBound46971
def owner : Owner := ⟨.program ⟨214⟩, ⟨29625⟩⟩
def transferEvent : Nat := 46971
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46969 .coefficient) (.predecessor 1 46970 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46969 .coefficient)
      LeftBound46964.bound (LeftBound46964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events183.exact46968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46964.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46970 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46964.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46964.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46964.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46971

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
