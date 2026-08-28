import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard187
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard188

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound28429
def owner : Owner := ⟨.program ⟨214⟩, ⟨25315⟩⟩
def transferEvent : Nat := 28429
def frameStart : Nat := 28315
def rule : BoundRule := .sum [.predecessor 0 28427 .coefficient, .predecessor 1 28428 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28427 .coefficient)
      LeftBound28425.bound (LeftBound28425.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28425.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28425.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28428 .coefficient)
      LeftBound28406.bound (LeftBound28406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28406.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28406.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28425.bound, LeftBound28406.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28425.bound, LeftBound28406.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28425.actual selector witness, LeftBound28406.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28429

namespace LeftBound28442
def owner : Owner := ⟨.program ⟨214⟩, ⟨25313⟩⟩
def transferEvent : Nat := 28442
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 28440 .coefficient, .predecessor 1 28441 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28440 .coefficient)
      LeftBound28263.bound (LeftBound28263.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28263.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28263.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28441 .coefficient)
      LeftBound28246.bound (LeftBound28246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28253RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28263.bound, LeftBound28246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28263.bound, LeftBound28246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28263.actual selector witness, LeftBound28246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28442

namespace LeftBound28445
def owner : Owner := ⟨.program ⟨214⟩, ⟨25313⟩⟩
def transferEvent : Nat := 28445
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 28439 .summary, .result 28253 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28439 .summary)
      LeftBound28265.bound (LeftBound28265.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19255⟩⟩) (rawTerms := some (Proof.Events111.exact28439RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28253 .summary)
      LeftBound28248.bound (LeftBound28248.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25312⟩⟩) (rawTerms := some (Proof.Events110.exact28253RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28248.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28265.bound, LeftBound28248.bound]
def bound : CoeffClass := .finite ⟨352024077676544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28265.bound, LeftBound28248.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28265.actual selector witness, LeftBound28248.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28445

namespace LeftBound28449
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def transferEvent : Nat := 28449
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28447 .coefficient) (.predecessor 1 28448 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28447 .coefficient)
      LeftBound28442.bound (LeftBound28442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28442.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28442.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28448 .coefficient)
      LeftAuthority28168.bound (LeftAuthority28168.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events110.exact28169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28168.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28442.bound LeftAuthority28168.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28442.bound, LeftAuthority28168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28442.actual selector witness) * (LeftAuthority28168.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28449

namespace LeftBound28450
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def transferEvent : Nat := 28450
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27037⟩⟩]⟩ [⟨.result 28169 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28169 .coefficient)
      LeftAuthority28168.bound (LeftAuthority28168.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27037⟩⟩) (rawTerms := some (Proof.Events110.exact28169RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28168.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28168.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28168.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28168.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28168.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28450

namespace LeftBound28451
def owner : Owner := ⟨.program ⟨214⟩, ⟨27039⟩⟩
def transferEvent : Nat := 28451
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 28446 .summary) (.transfer 28450) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28446 .summary)
      LeftBound28445.bound (LeftBound28445.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25313⟩⟩) (rawTerms := some (Proof.Events111.exact28446RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound28445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28450)
      LeftBound28450.bound (LeftBound28450.actual selector witness) := by
  exact .transfer (LeftBound28450.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28445.bound LeftBound28450.bound
def bound : CoeffClass := .finite ⟨1291933997458159304704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28445.bound, LeftBound28450.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28445.actual selector witness) * (LeftBound28450.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28451

namespace LeftBound28462
def owner : Owner := ⟨.program ⟨214⟩, ⟨20838⟩⟩
def transferEvent : Nat := 28462
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 28460 .coefficient) (.value (.predecessor 1 28461 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28460 .coefficient)
      LeftAuthority28458.bound (LeftAuthority28458.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28459RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28458.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28461 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority28458.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28458.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28458.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound28462

namespace LeftBound28466
def owner : Owner := ⟨.program ⟨214⟩, ⟨20839⟩⟩
def transferEvent : Nat := 28466
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 28464 .coefficient) (.predecessor 1 28465 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28464 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28465 .coefficient)
      LeftBound28462.bound (LeftBound28462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28463RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28462.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound28462.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound28462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound28462.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28466

namespace LeftBound28467
def owner : Owner := ⟨.program ⟨214⟩, ⟨20839⟩⟩
def transferEvent : Nat := 28467
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20836⟩⟩]⟩ [⟨.result 28459 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 28459 .coefficient)
      LeftAuthority28458.bound (LeftAuthority28458.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20836⟩⟩) (rawTerms := some (Proof.Events111.exact28459RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28458.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28458.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority28458.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority28458.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound28467

namespace LeftBound28468
def owner : Owner := ⟨.program ⟨214⟩, ⟨20839⟩⟩
def transferEvent : Nat := 28468
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 28467) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 28467)
      LeftBound28467.bound (LeftBound28467.actual selector witness) := by
  exact .transfer (LeftBound28467.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound28467.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound28467.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound28467.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28468

namespace LeftBound28563
def owner : Owner := ⟨.program ⟨214⟩, ⟨15435⟩⟩
def transferEvent : Nat := 28563
def frameStart : Nat := 28524
def rule : BoundRule := .identity (.predecessor 0 28562 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28562 .coefficient)
      LeftAuthority28560.bound (LeftAuthority28560.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28560.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28560.derived selector witness)

def rawBound : CoeffClass := LeftAuthority28560.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority28560.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28563

namespace LeftBound28580
def owner : Owner := ⟨.program ⟨214⟩, ⟨15474⟩⟩
def transferEvent : Nat := 28580
def frameStart : Nat := 28524
def rule : BoundRule := .sum [.predecessor 0 28578 .coefficient, .predecessor 1 28579 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28578 .coefficient)
      LeftBound28563.bound (LeftBound28563.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28579 .coefficient)
      LeftAuthority28576.bound (LeftAuthority28576.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority28576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound28563.bound, LeftAuthority28576.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28563.bound, LeftAuthority28576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound28563.actual selector witness, LeftAuthority28576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28580

namespace LeftBound28583
def owner : Owner := ⟨.program ⟨214⟩, ⟨15475⟩⟩
def transferEvent : Nat := 28583
def frameStart : Nat := 28524
def rule : BoundRule := .identity (.predecessor 0 28582 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28582 .coefficient)
      LeftBound28580.bound (LeftBound28580.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound28580.derived selector witness)

def rawBound : CoeffClass := LeftBound28580.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28580.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound28580.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound28583

namespace LeftBound28589
def owner : Owner := ⟨.program ⟨214⟩, ⟨15476⟩⟩
def transferEvent : Nat := 28589
def frameStart : Nat := 28524
def rule : BoundRule := .product (.predecessor 0 28587 .coefficient) (.predecessor 1 28588 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28587 .coefficient)
      LeftAuthority28585.bound (LeftAuthority28585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28588 .coefficient)
      LeftBound28583.bound (LeftBound28583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28583.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority28585.bound LeftBound28583.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28585.bound, LeftBound28583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority28585.actual selector witness) * (LeftBound28583.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28589

namespace LeftBound28597
def owner : Owner := ⟨.program ⟨214⟩, ⟨15477⟩⟩
def transferEvent : Nat := 28597
def frameStart : Nat := 28524
def rule : BoundRule := .sum [.predecessor 0 28595 .coefficient, .predecessor 1 28596 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28595 .coefficient)
      LeftAuthority28593.bound (LeftAuthority28593.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28594RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28593.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28593.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28596 .coefficient)
      LeftBound28589.bound (LeftBound28589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28589.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority28593.bound, LeftBound28589.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority28593.bound, LeftBound28589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority28593.actual selector witness, LeftBound28589.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound28597

namespace LeftBound28601
def owner : Owner := ⟨.program ⟨214⟩, ⟨27038⟩⟩
def transferEvent : Nat := 28601
def frameStart : Nat := 28524
def rule : BoundRule := .product (.predecessor 0 28599 .coefficient) (.predecessor 1 28600 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 28599 .coefficient)
      LeftBound28597.bound (LeftBound28597.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound28597.bound, RecordedBoundRefines] <;> decide)
      (LeftBound28597.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 28600 .coefficient)
      LeftAuthority28574.bound (LeftAuthority28574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events111.exact28575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority28574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority28574.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound28597.bound LeftAuthority28574.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound28597.bound, LeftAuthority28574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound28597.actual selector witness) * (LeftAuthority28574.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound28601

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
