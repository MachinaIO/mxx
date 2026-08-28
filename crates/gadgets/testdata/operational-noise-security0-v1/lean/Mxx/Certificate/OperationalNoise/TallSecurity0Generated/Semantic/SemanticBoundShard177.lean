import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard176

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26801
def owner : Owner := ⟨.program ⟨214⟩, ⟨26005⟩⟩
def transferEvent : Nat := 26801
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26004⟩⟩]⟩ [⟨.result 26733 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26733 .coefficient)
      LeftAuthority26732.bound (LeftAuthority26732.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26004⟩⟩) (rawTerms := some (Proof.Events104.exact26733RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26732.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26732.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26732.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26732.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26732.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26801

namespace LeftBound26802
def owner : Owner := ⟨.program ⟨214⟩, ⟨26005⟩⟩
def transferEvent : Nat := 26802
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26797 .summary) (.transfer 26801) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26797 .summary)
      LeftBound26796.bound (LeftBound26796.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14025⟩⟩) (rawTerms := some (Proof.Events104.exact26797RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26796.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26801)
      LeftBound26801.bound (LeftBound26801.actual selector witness) := by
  exact .transfer (LeftBound26801.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26796.bound LeftBound26801.bound
def bound : CoeffClass := .finite ⟨350243308699648, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26796.bound, LeftBound26801.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26796.actual selector witness) * (LeftBound26801.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26802

namespace LeftBound26813
def owner : Owner := ⟨.program ⟨214⟩, ⟨19470⟩⟩
def transferEvent : Nat := 26813
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 26811 .coefficient) (.value (.predecessor 1 26812 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26811 .coefficient)
      LeftAuthority26809.bound (LeftAuthority26809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26812 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26809.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26809.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26809.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26813

namespace LeftBound26817
def owner : Owner := ⟨.program ⟨214⟩, ⟨19471⟩⟩
def transferEvent : Nat := 26817
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26815 .coefficient) (.predecessor 1 26816 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26815 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26816 .coefficient)
      LeftBound26813.bound (LeftBound26813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events104.exact26814RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26813.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound26813.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound26813.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound26813.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26817

namespace LeftBound26818
def owner : Owner := ⟨.program ⟨214⟩, ⟨19471⟩⟩
def transferEvent : Nat := 26818
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19468⟩⟩]⟩ [⟨.result 26810 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 26810 .coefficient)
      LeftAuthority26809.bound (LeftAuthority26809.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19468⟩⟩) (rawTerms := some (Proof.Events104.exact26810RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26809.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26809.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26809.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26809.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26818

namespace LeftBound26819
def owner : Owner := ⟨.program ⟨214⟩, ⟨19471⟩⟩
def transferEvent : Nat := 26819
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 26818) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 26818)
      LeftBound26818.bound (LeftBound26818.actual selector witness) := by
  exact .transfer (LeftBound26818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound26818.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound26818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound26818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26819

namespace LeftBound26898
def owner : Owner := ⟨.program ⟨214⟩, ⟨14018⟩⟩
def transferEvent : Nat := 26898
def frameStart : Nat := 26869
def rule : BoundRule := .product (.predecessor 0 26896 .coefficient) (.predecessor 1 26897 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26896 .coefficient)
      LeftAuthority26894.bound (LeftAuthority26894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26897 .coefficient)
      LeftAuthority26891.bound (LeftAuthority26891.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26891.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26891.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority26894.bound LeftAuthority26891.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26894.bound, LeftAuthority26891.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority26894.actual selector witness) * (LeftAuthority26891.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26898

namespace LeftBound26902
def owner : Owner := ⟨.program ⟨214⟩, ⟨14019⟩⟩
def transferEvent : Nat := 26902
def frameStart : Nat := 26869
def rule : BoundRule := .identity (.predecessor 0 26901 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26901 .coefficient)
      LeftBound26898.bound (LeftBound26898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26898.derived selector witness)

def rawBound : CoeffClass := LeftBound26898.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound26898.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26902

namespace LeftBound26919
def owner : Owner := ⟨.program ⟨214⟩, ⟨14109⟩⟩
def transferEvent : Nat := 26919
def frameStart : Nat := 26869
def rule : BoundRule := .sum [.predecessor 0 26917 .coefficient, .predecessor 1 26918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26917 .coefficient)
      LeftBound26902.bound (LeftBound26902.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26918 .coefficient)
      LeftAuthority26915.bound (LeftAuthority26915.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority26915.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26902.bound, LeftAuthority26915.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26902.bound, LeftAuthority26915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26902.actual selector witness, LeftAuthority26915.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26919

namespace LeftBound26922
def owner : Owner := ⟨.program ⟨214⟩, ⟨14110⟩⟩
def transferEvent : Nat := 26922
def frameStart : Nat := 26869
def rule : BoundRule := .identity (.predecessor 0 26921 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26921 .coefficient)
      LeftBound26919.bound (LeftBound26919.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound26919.derived selector witness)

def rawBound : CoeffClass := LeftBound26919.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound26919.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26922

namespace LeftBound26928
def owner : Owner := ⟨.program ⟨214⟩, ⟨14111⟩⟩
def transferEvent : Nat := 26928
def frameStart : Nat := 26869
def rule : BoundRule := .product (.predecessor 0 26926 .coefficient) (.predecessor 1 26927 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26926 .coefficient)
      LeftAuthority26924.bound (LeftAuthority26924.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26925RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26924.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26924.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26927 .coefficient)
      LeftBound26922.bound (LeftBound26922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26922.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority26924.bound LeftBound26922.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26924.bound, LeftBound26922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority26924.actual selector witness) * (LeftBound26922.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26928

namespace LeftBound26944
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 26944
def frameStart : Nat := 26869
def rule : BoundRule := .scale (.predecessor 0 26942 .coefficient) (.value (.predecessor 1 26943 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26942 .coefficient)
      LeftAuthority26940.bound (LeftAuthority26940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26940.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26943 .coefficient)
      LeftAuthority26931.bound (LeftAuthority26931.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority26931.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26940.bound LeftAuthority26931.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26940.bound, LeftAuthority26931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority26940.actual selector witness) * (LeftAuthority26931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26944

namespace LeftBound26947
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 26947
def frameStart : Nat := 26869
def rule : BoundRule := .identity (.predecessor 0 26946 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26946 .coefficient)
      LeftAuthority26934.bound (LeftAuthority26934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26934.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26934.derived selector witness)

def rawBound : CoeffClass := LeftAuthority26934.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority26934.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound26947

namespace LeftBound26951
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 26951
def frameStart : Nat := 26869
def rule : BoundRule := .product (.predecessor 0 26949 .coefficient) (.predecessor 1 26950 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26949 .coefficient)
      LeftBound26947.bound (LeftBound26947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26950 .coefficient)
      LeftBound26944.bound (LeftBound26944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26944.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26947.bound LeftBound26944.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26947.bound, LeftBound26944.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26947.actual selector witness) * (LeftBound26944.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26951

namespace LeftBound26956
def owner : Owner := ⟨.program ⟨214⟩, ⟨14112⟩⟩
def transferEvent : Nat := 26956
def frameStart : Nat := 26869
def rule : BoundRule := .sum [.predecessor 0 26954 .coefficient, .predecessor 1 26955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26954 .coefficient)
      LeftBound26951.bound (LeftBound26951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26955 .coefficient)
      LeftBound26928.bound (LeftBound26928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26928.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26928.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound26951.bound, LeftBound26928.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26951.bound, LeftBound26928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound26951.actual selector witness, LeftBound26928.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound26956

namespace LeftBound26960
def owner : Owner := ⟨.program ⟨214⟩, ⟨26007⟩⟩
def transferEvent : Nat := 26960
def frameStart : Nat := 26869
def rule : BoundRule := .product (.predecessor 0 26958 .coefficient) (.predecessor 1 26959 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 26958 .coefficient)
      LeftBound26956.bound (LeftBound26956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 26959 .coefficient)
      LeftAuthority26913.bound (LeftAuthority26913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact26914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26913.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound26956.bound LeftAuthority26913.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26956.bound, LeftAuthority26913.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound26956.actual selector witness) * (LeftAuthority26913.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26960

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
