import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard189
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard224

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound34679
def owner : Owner := ⟨.program ⟨214⟩, ⟨27251⟩⟩
def transferEvent : Nat := 34679
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
end LeftBound34679

namespace LeftBound34680
def owner : Owner := ⟨.program ⟨214⟩, ⟨27251⟩⟩
def transferEvent : Nat := 34680
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 34675 .summary) (.transfer 34679) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34675 .summary)
      LeftBound34674.bound (LeftBound34674.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27250⟩⟩) (rawTerms := some (Proof.Events135.exact34675RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound34674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34679)
      LeftBound34679.bound (LeftBound34679.actual selector witness) := by
  exact .transfer (LeftBound34679.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34674.bound LeftBound34679.bound
def bound : CoeffClass := .finite ⟨4741582956326566183208747008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34674.bound, LeftBound34679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34674.actual selector witness) * (LeftBound34679.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34680

namespace LeftBound34695
def owner : Owner := ⟨.program ⟨214⟩, ⟨27032⟩⟩
def transferEvent : Nat := 34695
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34693 .coefficient) (.predecessor 1 34694 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34693 .coefficient)
      LeftBound28442.bound (LeftBound28442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34694 .coefficient)
      LeftAuthority34691.bound (LeftAuthority34691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34691.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28442.bound LeftAuthority34691.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28442.bound, LeftAuthority34691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28442.actual selector witness) * (LeftAuthority34691.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34695

namespace LeftBound34696
def owner : Owner := ⟨.program ⟨214⟩, ⟨27032⟩⟩
def transferEvent : Nat := 34696
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27030⟩⟩]⟩ [⟨.result 34692 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34692 .coefficient)
      LeftAuthority34691.bound (LeftAuthority34691.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27030⟩⟩) (rawTerms := some (Proof.Events135.exact34692RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34691.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34691.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34691.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34696

namespace LeftBound34697
def owner : Owner := ⟨.program ⟨214⟩, ⟨27032⟩⟩
def transferEvent : Nat := 34697
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28446 .summary) (.transfer 34696) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28446 .summary)
      LeftBound28445.bound (LeftBound28445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25313⟩⟩) (rawTerms := some (Proof.Events111.exact28446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34696)
      LeftBound34696.bound (LeftBound34696.actual selector witness) := by
  exact .transfer (LeftBound34696.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28445.bound LeftBound34696.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28445.bound, LeftBound34696.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28445.actual selector witness) * (LeftBound34696.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34697

namespace LeftBound34708
def owner : Owner := ⟨.program ⟨214⟩, ⟨20766⟩⟩
def transferEvent : Nat := 34708
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 34706 .coefficient) (.value (.predecessor 1 34707 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34706 .coefficient)
      LeftAuthority34704.bound (LeftAuthority34704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34707 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority34704.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34704.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34704.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound34708

namespace LeftBound34712
def owner : Owner := ⟨.program ⟨214⟩, ⟨20767⟩⟩
def transferEvent : Nat := 34712
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 34710 .coefficient) (.predecessor 1 34711 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34710 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34711 .coefficient)
      LeftBound34708.bound (LeftBound34708.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34709RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34708.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34708.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound34708.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound34708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound34708.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34712

namespace LeftBound34713
def owner : Owner := ⟨.program ⟨214⟩, ⟨20767⟩⟩
def transferEvent : Nat := 34713
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20764⟩⟩]⟩ [⟨.result 34705 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 34705 .coefficient)
      LeftAuthority34704.bound (LeftAuthority34704.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20764⟩⟩) (rawTerms := some (Proof.Events135.exact34705RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34704.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34704.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority34704.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34704.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority34704.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound34713

namespace LeftBound34714
def owner : Owner := ⟨.program ⟨214⟩, ⟨20767⟩⟩
def transferEvent : Nat := 34714
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 34713) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 34713)
      LeftBound34713.bound (LeftBound34713.actual selector witness) := by
  exact .transfer (LeftBound34713.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound34713.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound34713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound34713.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34714

namespace LeftBound34809
def owner : Owner := ⟨.program ⟨214⟩, ⟨15435⟩⟩
def transferEvent : Nat := 34809
def frameStart : Nat := 34770
def rule : BoundRule := .identity (.predecessor 0 34808 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34808 .coefficient)
      LeftAuthority34806.bound (LeftAuthority34806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events135.exact34807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34806.derived selector witness)

def rawBound : CoeffClass := LeftAuthority34806.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34806.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority34806.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34809

namespace LeftBound34826
def owner : Owner := ⟨.program ⟨214⟩, ⟨15474⟩⟩
def transferEvent : Nat := 34826
def frameStart : Nat := 34770
def rule : BoundRule := .sum [.predecessor 0 34824 .coefficient, .predecessor 1 34825 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34824 .coefficient)
      LeftBound34809.bound (LeftBound34809.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34809.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34825 .coefficient)
      LeftAuthority34822.bound (LeftAuthority34822.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority34822.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound34809.bound, LeftAuthority34822.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34809.bound, LeftAuthority34822.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound34809.actual selector witness, LeftAuthority34822.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34826

namespace LeftBound34829
def owner : Owner := ⟨.program ⟨214⟩, ⟨15475⟩⟩
def transferEvent : Nat := 34829
def frameStart : Nat := 34770
def rule : BoundRule := .identity (.predecessor 0 34828 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34828 .coefficient)
      LeftBound34826.bound (LeftBound34826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound34826.derived selector witness)

def rawBound : CoeffClass := LeftBound34826.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound34826.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound34829

namespace LeftBound34835
def owner : Owner := ⟨.program ⟨214⟩, ⟨15476⟩⟩
def transferEvent : Nat := 34835
def frameStart : Nat := 34770
def rule : BoundRule := .product (.predecessor 0 34833 .coefficient) (.predecessor 1 34834 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34833 .coefficient)
      LeftAuthority34831.bound (LeftAuthority34831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34834 .coefficient)
      LeftBound34829.bound (LeftBound34829.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34829.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34829.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority34831.bound LeftBound34829.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34831.bound, LeftBound34829.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority34831.actual selector witness) * (LeftBound34829.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34835

namespace LeftBound34843
def owner : Owner := ⟨.program ⟨214⟩, ⟨15477⟩⟩
def transferEvent : Nat := 34843
def frameStart : Nat := 34770
def rule : BoundRule := .sum [.predecessor 0 34841 .coefficient, .predecessor 1 34842 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34841 .coefficient)
      LeftAuthority34839.bound (LeftAuthority34839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34839.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34842 .coefficient)
      LeftBound34835.bound (LeftBound34835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34837RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority34839.bound, LeftBound34835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34839.bound, LeftBound34835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority34839.actual selector witness, LeftBound34835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound34843

namespace LeftBound34847
def owner : Owner := ⟨.program ⟨214⟩, ⟨27031⟩⟩
def transferEvent : Nat := 34847
def frameStart : Nat := 34770
def rule : BoundRule := .product (.predecessor 0 34845 .coefficient) (.predecessor 1 34846 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34845 .coefficient)
      LeftBound34843.bound (LeftBound34843.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound34843.bound, RecordedBoundRefines] <;> decide)
      (LeftBound34843.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34846 .coefficient)
      LeftAuthority34820.bound (LeftAuthority34820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34820.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound34843.bound LeftAuthority34820.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound34843.bound, LeftAuthority34820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound34843.actual selector witness) * (LeftAuthority34820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34847

namespace LeftBound34858
def owner : Owner := ⟨.program ⟨214⟩, ⟨15534⟩⟩
def transferEvent : Nat := 34858
def frameStart : Nat := 34770
def rule : BoundRule := .product (.predecessor 0 34856 .coefficient) (.predecessor 1 34857 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 34856 .coefficient)
      LeftAuthority34831.bound (LeftAuthority34831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34831.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 34857 .coefficient)
      LeftAuthority34854.bound (LeftAuthority34854.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events136.exact34855RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority34854.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority34854.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority34831.bound LeftAuthority34854.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority34831.bound, LeftAuthority34854.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority34831.actual selector witness) * (LeftAuthority34854.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound34858

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
