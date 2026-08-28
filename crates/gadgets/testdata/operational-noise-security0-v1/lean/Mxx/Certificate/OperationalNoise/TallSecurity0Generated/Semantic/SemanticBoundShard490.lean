import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard489

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound71844
def owner : Owner := ⟨.program ⟨214⟩, ⟨27204⟩⟩
def transferEvent : Nat := 71844
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 71839 .summary) (.transfer 71843) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71839 .summary)
      LeftBound71838.bound (LeftBound71838.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25832⟩⟩) (rawTerms := some (Proof.Events280.exact71839RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71838.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71843)
      LeftBound71843.bound (LeftBound71843.actual selector witness) := by
  exact .transfer (LeftBound71843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71838.bound LeftBound71843.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71838.bound, LeftBound71843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71838.actual selector witness) * (LeftBound71843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71844

namespace LeftBound71855
def owner : Owner := ⟨.program ⟨214⟩, ⟨20966⟩⟩
def transferEvent : Nat := 71855
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 71853 .coefficient) (.value (.predecessor 1 71854 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71853 .coefficient)
      LeftAuthority71851.bound (LeftAuthority71851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71854 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority71851.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71851.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71851.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound71855

namespace LeftBound71859
def owner : Owner := ⟨.program ⟨214⟩, ⟨20967⟩⟩
def transferEvent : Nat := 71859
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 71857 .coefficient) (.predecessor 1 71858 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71857 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71858 .coefficient)
      LeftBound71855.bound (LeftBound71855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound71855.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound71855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound71855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71859

namespace LeftBound71860
def owner : Owner := ⟨.program ⟨214⟩, ⟨20967⟩⟩
def transferEvent : Nat := 71860
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20964⟩⟩]⟩ [⟨.result 71852 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71852 .coefficient)
      LeftAuthority71851.bound (LeftAuthority71851.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20964⟩⟩) (rawTerms := some (Proof.Events280.exact71852RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71851.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority71851.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority71851.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound71860

namespace LeftBound71861
def owner : Owner := ⟨.program ⟨214⟩, ⟨20967⟩⟩
def transferEvent : Nat := 71861
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 71860) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 71860)
      LeftBound71860.bound (LeftBound71860.actual selector witness) := by
  exact .transfer (LeftBound71860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound71860.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound71860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound71860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71861

namespace LeftBound71956
def owner : Owner := ⟨.program ⟨214⟩, ⟨15580⟩⟩
def transferEvent : Nat := 71956
def frameStart : Nat := 71917
def rule : BoundRule := .identity (.predecessor 0 71955 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71955 .coefficient)
      LeftAuthority71953.bound (LeftAuthority71953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71953.derived selector witness)

def rawBound : CoeffClass := LeftAuthority71953.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority71953.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71956

namespace LeftBound71973
def owner : Owner := ⟨.program ⟨214⟩, ⟨15654⟩⟩
def transferEvent : Nat := 71973
def frameStart : Nat := 71917
def rule : BoundRule := .sum [.predecessor 0 71971 .coefficient, .predecessor 1 71972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71971 .coefficient)
      LeftBound71956.bound (LeftBound71956.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71972 .coefficient)
      LeftAuthority71969.bound (LeftAuthority71969.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority71969.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71956.bound, LeftAuthority71969.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71956.bound, LeftAuthority71969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71956.actual selector witness, LeftAuthority71969.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71973

namespace LeftBound71976
def owner : Owner := ⟨.program ⟨214⟩, ⟨15655⟩⟩
def transferEvent : Nat := 71976
def frameStart : Nat := 71917
def rule : BoundRule := .identity (.predecessor 0 71975 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71975 .coefficient)
      LeftBound71973.bound (LeftBound71973.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound71973.derived selector witness)

def rawBound : CoeffClass := LeftBound71973.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound71973.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound71976

namespace LeftBound71982
def owner : Owner := ⟨.program ⟨214⟩, ⟨15656⟩⟩
def transferEvent : Nat := 71982
def frameStart : Nat := 71917
def rule : BoundRule := .product (.predecessor 0 71980 .coefficient) (.predecessor 1 71981 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71980 .coefficient)
      LeftAuthority71978.bound (LeftAuthority71978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71981 .coefficient)
      LeftBound71976.bound (LeftBound71976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority71978.bound LeftBound71976.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71978.bound, LeftBound71976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority71978.actual selector witness) * (LeftBound71976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71982

namespace LeftBound71990
def owner : Owner := ⟨.program ⟨214⟩, ⟨15657⟩⟩
def transferEvent : Nat := 71990
def frameStart : Nat := 71917
def rule : BoundRule := .sum [.predecessor 0 71988 .coefficient, .predecessor 1 71989 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71988 .coefficient)
      LeftAuthority71986.bound (LeftAuthority71986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71986.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71989 .coefficient)
      LeftBound71982.bound (LeftBound71982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71984RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71982.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority71986.bound, LeftBound71982.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71986.bound, LeftBound71982.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority71986.actual selector witness, LeftBound71982.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound71990

namespace LeftBound71994
def owner : Owner := ⟨.program ⟨214⟩, ⟨27203⟩⟩
def transferEvent : Nat := 71994
def frameStart : Nat := 71917
def rule : BoundRule := .product (.predecessor 0 71992 .coefficient) (.predecessor 1 71993 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 71992 .coefficient)
      LeftBound71990.bound (LeftBound71990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 71993 .coefficient)
      LeftAuthority71967.bound (LeftAuthority71967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71967.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound71990.bound LeftAuthority71967.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71990.bound, LeftAuthority71967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound71990.actual selector witness) * (LeftAuthority71967.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound71994

namespace LeftBound72005
def owner : Owner := ⟨.program ⟨214⟩, ⟨15627⟩⟩
def transferEvent : Nat := 72005
def frameStart : Nat := 71917
def rule : BoundRule := .product (.predecessor 0 72003 .coefficient) (.predecessor 1 72004 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72003 .coefficient)
      LeftAuthority71978.bound (LeftAuthority71978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority71978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority71978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72004 .coefficient)
      LeftAuthority72001.bound (LeftAuthority72001.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72002RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72001.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72001.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority71978.bound LeftAuthority72001.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority71978.bound, LeftAuthority72001.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority71978.actual selector witness) * (LeftAuthority72001.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72005

namespace LeftBound72013
def owner : Owner := ⟨.program ⟨214⟩, ⟨15628⟩⟩
def transferEvent : Nat := 72013
def frameStart : Nat := 71917
def rule : BoundRule := .sum [.predecessor 0 72011 .coefficient, .predecessor 1 72012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72011 .coefficient)
      LeftAuthority72009.bound (LeftAuthority72009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72012 .coefficient)
      LeftBound72005.bound (LeftBound72005.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72005.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72009.bound, LeftBound72005.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72009.bound, LeftBound72005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72009.actual selector witness, LeftBound72005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72013

namespace LeftBound72017
def owner : Owner := ⟨.program ⟨214⟩, ⟨27207⟩⟩
def transferEvent : Nat := 72017
def frameStart : Nat := 71917
def rule : BoundRule := .sum [.predecessor 0 72015 .coefficient, .predecessor 1 72016 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72015 .coefficient)
      LeftBound72013.bound (LeftBound72013.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72013.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72013.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72016 .coefficient)
      LeftBound71994.bound (LeftBound71994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact71999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71994.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72013.bound, LeftBound71994.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72013.bound, LeftBound71994.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72013.actual selector witness, LeftBound71994.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72017

namespace LeftBound72030
def owner : Owner := ⟨.program ⟨214⟩, ⟨27205⟩⟩
def transferEvent : Nat := 72030
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72028 .coefficient, .predecessor 1 72029 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72028 .coefficient)
      LeftBound71859.bound (LeftBound71859.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71859.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71859.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72029 .coefficient)
      LeftBound71842.bound (LeftBound71842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events280.exact71849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound71842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound71842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71859.bound, LeftBound71842.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71859.bound, LeftBound71842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71859.actual selector witness, LeftBound71842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72030

namespace LeftBound72033
def owner : Owner := ⟨.program ⟨214⟩, ⟨27205⟩⟩
def transferEvent : Nat := 72033
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72027 .summary, .result 71849 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72027 .summary)
      LeftBound71861.bound (LeftBound71861.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20967⟩⟩) (rawTerms := some (Proof.Events281.exact72027RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 71849 .summary)
      LeftBound71844.bound (LeftBound71844.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27204⟩⟩) (rawTerms := some (Proof.Events280.exact71849RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound71844.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound71861.bound, LeftBound71844.bound]
def bound : CoeffClass := .finite ⟨1291978824159503986688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound71861.bound, LeftBound71844.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound71861.actual selector witness, LeftBound71844.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72033

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
