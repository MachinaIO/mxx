import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard080
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard686

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound99974
def owner : Owner := ⟨.program ⟨214⟩, ⟨15742⟩⟩
def transferEvent : Nat := 99974
def frameStart : Nat := 99898
def rule : BoundRule := .product (.predecessor 0 99972 .coefficient) (.predecessor 1 99973 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99972 .coefficient)
      LeftAuthority99947.bound (LeftAuthority99947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99973 .coefficient)
      LeftAuthority99970.bound (LeftAuthority99970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99970.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority99947.bound LeftAuthority99970.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99947.bound, LeftAuthority99970.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority99947.actual selector witness) * (LeftAuthority99970.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound99974

namespace LeftBound99982
def owner : Owner := ⟨.program ⟨214⟩, ⟨15743⟩⟩
def transferEvent : Nat := 99982
def frameStart : Nat := 99898
def rule : BoundRule := .sum [.predecessor 0 99980 .coefficient, .predecessor 1 99981 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99980 .coefficient)
      LeftAuthority99978.bound (LeftAuthority99978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority99978.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority99978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99981 .coefficient)
      LeftBound99974.bound (LeftBound99974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99974.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority99978.bound, LeftBound99974.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority99978.bound, LeftBound99974.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority99978.actual selector witness, LeftBound99974.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99982

namespace LeftBound99986
def owner : Owner := ⟨.program ⟨214⟩, ⟨27402⟩⟩
def transferEvent : Nat := 99986
def frameStart : Nat := 99898
def rule : BoundRule := .sum [.predecessor 0 99984 .coefficient, .predecessor 1 99985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99984 .coefficient)
      LeftBound99982.bound (LeftBound99982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99985 .coefficient)
      LeftBound99963.bound (LeftBound99963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99963.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99982.bound, LeftBound99963.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99982.bound, LeftBound99963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99982.actual selector witness, LeftBound99963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99986

namespace LeftBound99999
def owner : Owner := ⟨.program ⟨214⟩, ⟨27400⟩⟩
def transferEvent : Nat := 99999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 99997 .coefficient, .predecessor 1 99998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 99997 .coefficient)
      LeftBound99852.bound (LeftBound99852.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99852.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99852.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 99998 .coefficient)
      LeftBound99835.bound (LeftBound99835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact99842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound99835.bound, RecordedBoundRefines] <;> decide)
      (LeftBound99835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99852.bound, LeftBound99835.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99852.bound, LeftBound99835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99852.actual selector witness, LeftBound99835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound99999

namespace LeftBound100002
def owner : Owner := ⟨.program ⟨214⟩, ⟨27400⟩⟩
def transferEvent : Nat := 100002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 99996 .summary, .result 99842 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99996 .summary)
      LeftBound99854.bound (LeftBound99854.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21104⟩⟩) (rawTerms := some (Proof.Events390.exact99996RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99854.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 99842 .summary)
      LeftBound99837.bound (LeftBound99837.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27399⟩⟩) (rawTerms := some (Proof.Events390.exact99842RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound99837.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound99854.bound, LeftBound99837.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound99854.bound, LeftBound99837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound99854.actual selector witness, LeftBound99837.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100002

namespace LeftBound100026
def owner : Owner := ⟨.program ⟨214⟩, ⟨11206⟩⟩
def transferEvent : Nat := 100026
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 100024 .coefficient) (.predecessor 1 100025 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100024 .coefficient)
      LeftAuthority4864.bound (LeftAuthority4864.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4864.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100025 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4864.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4864.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4864.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100026

namespace LeftBound100031
def owner : Owner := ⟨.program ⟨214⟩, ⟨7113⟩⟩
def transferEvent : Nat := 100031
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100029 .coefficient) (.predecessor 1 100030 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100029 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100030 .coefficient)
      LeftBound12984.bound (LeftBound12984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12984.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound12984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound12984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound12984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100031

namespace LeftBound100036
def owner : Owner := ⟨.program ⟨214⟩, ⟨11207⟩⟩
def transferEvent : Nat := 100036
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100034 .coefficient, .predecessor 1 100035 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100034 .coefficient)
      LeftBound100031.bound (LeftBound100031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100035 .coefficient)
      LeftBound100026.bound (LeftBound100026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100031.bound, LeftBound100026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100031.bound, LeftBound100026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100031.actual selector witness, LeftBound100026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100036

namespace LeftBound100040
def owner : Owner := ⟨.program ⟨214⟩, ⟨11208⟩⟩
def transferEvent : Nat := 100040
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100038 .coefficient, .predecessor 1 100039 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100038 .coefficient)
      LeftBound100036.bound (LeftBound100036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100037RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100039 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100036.bound, LeftBound12976.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100036.bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100036.actual selector witness, LeftBound12976.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100040

namespace LeftBound100041
def owner : Owner := ⟨.program ⟨214⟩, ⟨11208⟩⟩
def transferEvent : Nat := 100041
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨90⟩⟩]⟩ [⟨.result 12977 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12977 .coefficient)
      LeftBound12976.bound (LeftBound12976.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨90⟩⟩) (rawTerms := some (Proof.Events050.exact12977RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12976.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12976.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound12976.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound12976.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100041

namespace LeftBound100046
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def transferEvent : Nat := 100046
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100044 .coefficient) (.predecessor 1 100045 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100044 .coefficient)
      LeftBound100040.bound (LeftBound100040.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100040.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100040.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100045 .coefficient)
      LeftAuthority4867.bound (LeftAuthority4867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4867.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound100040.bound LeftAuthority4867.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100040.bound, LeftAuthority4867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound100040.actual selector witness) * (LeftAuthority4867.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100046

namespace LeftBound100047
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def transferEvent : Nat := 100047
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨13529⟩⟩], []⟩ [⟨.result 4868 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 4868 .coefficient)
      LeftAuthority4867.bound (LeftAuthority4867.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨13529⟩⟩) (rawTerms := some (Proof.Events019.exact4868RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4867.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority4867.bound []
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority4867.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100047

namespace LeftBound100048
def owner : Owner := ⟨.program ⟨214⟩, ⟨13532⟩⟩
def transferEvent : Nat := 100048
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 100043 .summary) (.transfer 100047) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100043 .summary)
      LeftBound100041.bound (LeftBound100041.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11208⟩⟩) (rawTerms := some (Proof.Events390.exact100043RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100047)
      LeftBound100047.bound (LeftBound100047.actual selector witness) := by
  exact .transfer (LeftBound100047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound100041.bound LeftBound100047.bound
def bound : CoeffClass := .finite ⟨8320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100041.bound, LeftBound100047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound100041.actual selector witness) * (LeftBound100047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100048

namespace LeftBound100054
def owner : Owner := ⟨.program ⟨214⟩, ⟨13533⟩⟩
def transferEvent : Nat := 100054
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 100052 .coefficient) (.predecessor 1 100053 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100052 .coefficient)
      LeftAuthority4867.bound (LeftAuthority4867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events019.exact4868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4867.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100053 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4867.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4867.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4867.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound100054

namespace LeftBound100059
def owner : Owner := ⟨.program ⟨214⟩, ⟨7130⟩⟩
def transferEvent : Nat := 100059
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 100057 .coefficient) (.predecessor 1 100058 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100057 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100058 .coefficient)
      LeftBound13025.bound (LeftBound13025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events050.exact13026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13025.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftBound13025.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound13025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftBound13025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100059

namespace LeftBound100064
def owner : Owner := ⟨.program ⟨214⟩, ⟨13534⟩⟩
def transferEvent : Nat := 100064
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 100062 .coefficient, .predecessor 1 100063 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 100062 .coefficient)
      LeftBound100059.bound (LeftBound100059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 100063 .coefficient)
      LeftBound100054.bound (LeftBound100054.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events390.exact100056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100054.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100054.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100059.bound, LeftBound100054.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100059.bound, LeftBound100054.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100059.actual selector witness, LeftBound100054.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound100064

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
