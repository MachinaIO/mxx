import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard283
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard323

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48897
def owner : Owner := ⟨.program ⟨214⟩, ⟨27453⟩⟩
def transferEvent : Nat := 48897
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27451⟩⟩]⟩ [⟨.result 48893 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48893 .coefficient)
      LeftAuthority48892.bound (LeftAuthority48892.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27451⟩⟩) (rawTerms := some (Proof.Events190.exact48893RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48892.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48892.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48892.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48892.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48897

namespace LeftBound48898
def owner : Owner := ⟨.program ⟨214⟩, ⟨27453⟩⟩
def transferEvent : Nat := 48898
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 42107 .summary) (.transfer 48897) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 42107 .summary)
      LeftBound42106.bound (LeftBound42106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25924⟩⟩) (rawTerms := some (Proof.Events164.exact42107RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound42106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48897)
      LeftBound48897.bound (LeftBound48897.actual selector witness) := by
  exact .transfer (LeftBound48897.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound42106.bound LeftBound48897.bound
def bound : CoeffClass := .finite ⟨1292001234793221062656, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound42106.bound, LeftBound48897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound42106.actual selector witness) * (LeftBound48897.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48898

namespace LeftBound48909
def owner : Owner := ⟨.program ⟨214⟩, ⟨21050⟩⟩
def transferEvent : Nat := 48909
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48907 .coefficient) (.value (.predecessor 1 48908 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48907 .coefficient)
      LeftAuthority48905.bound (LeftAuthority48905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact48906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48905.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48908 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48905.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48905.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48905.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48909

namespace LeftBound48913
def owner : Owner := ⟨.program ⟨214⟩, ⟨21051⟩⟩
def transferEvent : Nat := 48913
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48911 .coefficient) (.predecessor 1 48912 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48911 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48912 .coefficient)
      LeftBound48909.bound (LeftBound48909.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact48910RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48909.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48909.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound48909.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound48909.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound48909.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48913

namespace LeftBound48914
def owner : Owner := ⟨.program ⟨214⟩, ⟨21051⟩⟩
def transferEvent : Nat := 48914
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21048⟩⟩]⟩ [⟨.result 48906 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48906 .coefficient)
      LeftAuthority48905.bound (LeftAuthority48905.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21048⟩⟩) (rawTerms := some (Proof.Events191.exact48906RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48905.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48905.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48905.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48914

namespace LeftBound48915
def owner : Owner := ⟨.program ⟨214⟩, ⟨21051⟩⟩
def transferEvent : Nat := 48915
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 48914) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48914)
      LeftBound48914.bound (LeftBound48914.actual selector witness) := by
  exact .transfer (LeftBound48914.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound48914.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound48914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound48914.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48915

namespace LeftBound49010
def owner : Owner := ⟨.program ⟨214⟩, ⟨15711⟩⟩
def transferEvent : Nat := 49010
def frameStart : Nat := 48971
def rule : BoundRule := .identity (.predecessor 0 49009 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49009 .coefficient)
      LeftAuthority49007.bound (LeftAuthority49007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49007.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49007.derived selector witness)

def rawBound : CoeffClass := LeftAuthority49007.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority49007.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49010

namespace LeftBound49027
def owner : Owner := ⟨.program ⟨214⟩, ⟨15785⟩⟩
def transferEvent : Nat := 49027
def frameStart : Nat := 48971
def rule : BoundRule := .sum [.predecessor 0 49025 .coefficient, .predecessor 1 49026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49025 .coefficient)
      LeftBound49010.bound (LeftBound49010.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49026 .coefficient)
      LeftAuthority49023.bound (LeftAuthority49023.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority49023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49010.bound, LeftAuthority49023.bound]
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49010.bound, LeftAuthority49023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49010.actual selector witness, LeftAuthority49023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49027

namespace LeftBound49030
def owner : Owner := ⟨.program ⟨214⟩, ⟨15786⟩⟩
def transferEvent : Nat := 49030
def frameStart : Nat := 48971
def rule : BoundRule := .identity (.predecessor 0 49029 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49029 .coefficient)
      LeftBound49027.bound (LeftBound49027.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49027.derived selector witness)

def rawBound : CoeffClass := LeftBound49027.bound
def bound : CoeffClass := .finite ⟨12, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound49027.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49030

namespace LeftBound49036
def owner : Owner := ⟨.program ⟨214⟩, ⟨15787⟩⟩
def transferEvent : Nat := 49036
def frameStart : Nat := 48971
def rule : BoundRule := .product (.predecessor 0 49034 .coefficient) (.predecessor 1 49035 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49034 .coefficient)
      LeftAuthority49032.bound (LeftAuthority49032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49035 .coefficient)
      LeftBound49030.bound (LeftBound49030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49030.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49030.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority49032.bound LeftBound49030.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49032.bound, LeftBound49030.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority49032.actual selector witness) * (LeftBound49030.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49036

namespace LeftBound49044
def owner : Owner := ⟨.program ⟨214⟩, ⟨15788⟩⟩
def transferEvent : Nat := 49044
def frameStart : Nat := 48971
def rule : BoundRule := .sum [.predecessor 0 49042 .coefficient, .predecessor 1 49043 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49042 .coefficient)
      LeftAuthority49040.bound (LeftAuthority49040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49041RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49040.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49043 .coefficient)
      LeftBound49036.bound (LeftBound49036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49040.bound, LeftBound49036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49040.bound, LeftBound49036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49040.actual selector witness, LeftBound49036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49044

namespace LeftBound49048
def owner : Owner := ⟨.program ⟨214⟩, ⟨27452⟩⟩
def transferEvent : Nat := 49048
def frameStart : Nat := 48971
def rule : BoundRule := .product (.predecessor 0 49046 .coefficient) (.predecessor 1 49047 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49046 .coefficient)
      LeftBound49044.bound (LeftBound49044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49045RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49047 .coefficient)
      LeftAuthority49021.bound (LeftAuthority49021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49044.bound LeftAuthority49021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49044.bound, LeftAuthority49021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49044.actual selector witness) * (LeftAuthority49021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49048

namespace LeftBound49059
def owner : Owner := ⟨.program ⟨214⟩, ⟨17448⟩⟩
def transferEvent : Nat := 49059
def frameStart : Nat := 48971
def rule : BoundRule := .product (.predecessor 0 49057 .coefficient) (.predecessor 1 49058 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49057 .coefficient)
      LeftAuthority49032.bound (LeftAuthority49032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49058 .coefficient)
      LeftAuthority49055.bound (LeftAuthority49055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49055.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49055.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority49032.bound LeftAuthority49055.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49032.bound, LeftAuthority49055.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority49032.actual selector witness) * (LeftAuthority49055.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49059

namespace LeftBound49067
def owner : Owner := ⟨.program ⟨214⟩, ⟨17449⟩⟩
def transferEvent : Nat := 49067
def frameStart : Nat := 48971
def rule : BoundRule := .sum [.predecessor 0 49065 .coefficient, .predecessor 1 49066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49065 .coefficient)
      LeftAuthority49063.bound (LeftAuthority49063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49063.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49066 .coefficient)
      LeftBound49059.bound (LeftBound49059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49063.bound, LeftBound49059.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49063.bound, LeftBound49059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49063.actual selector witness, LeftBound49059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49067

namespace LeftBound49071
def owner : Owner := ⟨.program ⟨214⟩, ⟨27457⟩⟩
def transferEvent : Nat := 49071
def frameStart : Nat := 48971
def rule : BoundRule := .sum [.predecessor 0 49069 .coefficient, .predecessor 1 49070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49069 .coefficient)
      LeftBound49067.bound (LeftBound49067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49070 .coefficient)
      LeftBound49048.bound (LeftBound49048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49048.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49048.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49067.bound, LeftBound49048.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49067.bound, LeftBound49048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49067.actual selector witness, LeftBound49048.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49071

namespace LeftBound49084
def owner : Owner := ⟨.program ⟨214⟩, ⟨27454⟩⟩
def transferEvent : Nat := 49084
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 49082 .coefficient, .predecessor 1 49083 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49082 .coefficient)
      LeftBound48913.bound (LeftBound48913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact49081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48913.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49083 .coefficient)
      LeftBound48896.bound (LeftBound48896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events191.exact48903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48896.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48913.bound, LeftBound48896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48913.bound, LeftBound48896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48913.actual selector witness, LeftBound48896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49084

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
