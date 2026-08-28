import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard155

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound24055
def owner : Owner := ⟨.program ⟨214⟩, ⟨6765⟩⟩
def transferEvent : Nat := 24055
def frameStart : Nat := 23977
def rule : BoundRule := .identity (.predecessor 0 24054 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24054 .coefficient)
      LeftAuthority24042.bound (LeftAuthority24042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24043RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24042.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24042.derived selector witness)

def rawBound : CoeffClass := LeftAuthority24042.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority24042.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound24055

namespace LeftBound24059
def owner : Owner := ⟨.program ⟨214⟩, ⟨7869⟩⟩
def transferEvent : Nat := 24059
def frameStart : Nat := 23977
def rule : BoundRule := .product (.predecessor 0 24057 .coefficient) (.predecessor 1 24058 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24057 .coefficient)
      LeftBound24055.bound (LeftBound24055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24058 .coefficient)
      LeftBound24052.bound (LeftBound24052.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24053RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24052.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24052.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24055.bound LeftBound24052.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24055.bound, LeftBound24052.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24055.actual selector witness) * (LeftBound24052.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24059

namespace LeftBound24064
def owner : Owner := ⟨.program ⟨214⟩, ⟨12481⟩⟩
def transferEvent : Nat := 24064
def frameStart : Nat := 23977
def rule : BoundRule := .sum [.predecessor 0 24062 .coefficient, .predecessor 1 24063 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24062 .coefficient)
      LeftBound24059.bound (LeftBound24059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24061RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24063 .coefficient)
      LeftBound24036.bound (LeftBound24036.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24036.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24036.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24059.bound, LeftBound24036.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24059.bound, LeftBound24036.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24059.actual selector witness, LeftBound24036.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24064

namespace LeftBound24068
def owner : Owner := ⟨.program ⟨214⟩, ⟨25391⟩⟩
def transferEvent : Nat := 24068
def frameStart : Nat := 23977
def rule : BoundRule := .product (.predecessor 0 24066 .coefficient) (.predecessor 1 24067 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24066 .coefficient)
      LeftBound24064.bound (LeftBound24064.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24064.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24064.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24067 .coefficient)
      LeftAuthority24021.bound (LeftAuthority24021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24021.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24021.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24064.bound LeftAuthority24021.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24064.bound, LeftAuthority24021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24064.actual selector witness) * (LeftAuthority24021.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24068

namespace LeftBound24079
def owner : Owner := ⟨.program ⟨214⟩, ⟨16479⟩⟩
def transferEvent : Nat := 24079
def frameStart : Nat := 23977
def rule : BoundRule := .product (.predecessor 0 24077 .coefficient) (.predecessor 1 24078 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24077 .coefficient)
      LeftAuthority24032.bound (LeftAuthority24032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact24033RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24032.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24078 .coefficient)
      LeftAuthority24075.bound (LeftAuthority24075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24075.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24075.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority24032.bound LeftAuthority24075.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24032.bound, LeftAuthority24075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority24032.actual selector witness) * (LeftAuthority24075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24079

namespace LeftBound24087
def owner : Owner := ⟨.program ⟨214⟩, ⟨16480⟩⟩
def transferEvent : Nat := 24087
def frameStart : Nat := 23977
def rule : BoundRule := .sum [.predecessor 0 24085 .coefficient, .predecessor 1 24086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24085 .coefficient)
      LeftAuthority24083.bound (LeftAuthority24083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24086 .coefficient)
      LeftBound24079.bound (LeftBound24079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority24083.bound, LeftBound24079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24083.bound, LeftBound24079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority24083.actual selector witness, LeftBound24079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24087

namespace LeftBound24091
def owner : Owner := ⟨.program ⟨214⟩, ⟨25392⟩⟩
def transferEvent : Nat := 24091
def frameStart : Nat := 23977
def rule : BoundRule := .sum [.predecessor 0 24089 .coefficient, .predecessor 1 24090 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24089 .coefficient)
      LeftBound24087.bound (LeftBound24087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24090 .coefficient)
      LeftBound24068.bound (LeftBound24068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24068.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24068.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound24087.bound, LeftBound24068.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24087.bound, LeftBound24068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound24087.actual selector witness, LeftBound24068.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24091

namespace LeftBound24104
def owner : Owner := ⟨.program ⟨214⟩, ⟨25390⟩⟩
def transferEvent : Nat := 24104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 24102 .coefficient, .predecessor 1 24103 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24102 .coefficient)
      LeftBound23925.bound (LeftBound23925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24103 .coefficient)
      LeftBound23908.bound (LeftBound23908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23925.bound, LeftBound23908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23925.bound, LeftBound23908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23925.actual selector witness, LeftBound23908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24104

namespace LeftBound24107
def owner : Owner := ⟨.program ⟨214⟩, ⟨25390⟩⟩
def transferEvent : Nat := 24107
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 24101 .summary, .result 23915 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24101 .summary)
      LeftBound23927.bound (LeftBound23927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19903⟩⟩) (rawTerms := some (Proof.Events094.exact24101RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23915 .summary)
      LeftBound23910.bound (LeftBound23910.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25389⟩⟩) (rawTerms := some (Proof.Events093.exact23915RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound23910.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23927.bound, LeftBound23910.bound]
def bound : CoeffClass := .finite ⟨352127895089152, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23927.bound, LeftBound23910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23927.actual selector witness, LeftBound23910.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound24107

namespace LeftBound24111
def owner : Owner := ⟨.program ⟨214⟩, ⟨28992⟩⟩
def transferEvent : Nat := 24111
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24109 .coefficient) (.predecessor 1 24110 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24109 .coefficient)
      LeftBound24104.bound (LeftBound24104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24110 .coefficient)
      LeftAuthority23830.bound (LeftAuthority23830.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events093.exact23831RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23830.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24104.bound LeftAuthority23830.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24104.bound, LeftAuthority23830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24104.actual selector witness) * (LeftAuthority23830.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24111

namespace LeftBound24112
def owner : Owner := ⟨.program ⟨214⟩, ⟨28992⟩⟩
def transferEvent : Nat := 24112
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28990⟩⟩]⟩ [⟨.result 23831 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 23831 .coefficient)
      LeftAuthority23830.bound (LeftAuthority23830.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28990⟩⟩) (rawTerms := some (Proof.Events093.exact23831RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23830.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23830.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority23830.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23830.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24112

namespace LeftBound24113
def owner : Owner := ⟨.program ⟨214⟩, ⟨28992⟩⟩
def transferEvent : Nat := 24113
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 24108 .summary) (.transfer 24112) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24108 .summary)
      LeftBound24107.bound (LeftBound24107.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25390⟩⟩) (rawTerms := some (Proof.Events094.exact24108RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound24107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24112)
      LeftBound24112.bound (LeftBound24112.actual selector witness) := by
  exact .transfer (LeftBound24112.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound24107.bound LeftBound24112.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24107.bound, LeftBound24112.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound24107.actual selector witness) * (LeftBound24112.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24113

namespace LeftBound24124
def owner : Owner := ⟨.program ⟨214⟩, ⟨22134⟩⟩
def transferEvent : Nat := 24124
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 24122 .coefficient) (.value (.predecessor 1 24123 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24122 .coefficient)
      LeftAuthority24120.bound (LeftAuthority24120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24120.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24123 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority24120.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24120.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24120.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound24124

namespace LeftBound24128
def owner : Owner := ⟨.program ⟨214⟩, ⟨22135⟩⟩
def transferEvent : Nat := 24128
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 24126 .coefficient) (.predecessor 1 24127 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 24126 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 24127 .coefficient)
      LeftBound24124.bound (LeftBound24124.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24125RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24124.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24124.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound24124.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound24124.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound24124.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24128

namespace LeftBound24129
def owner : Owner := ⟨.program ⟨214⟩, ⟨22135⟩⟩
def transferEvent : Nat := 24129
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22132⟩⟩]⟩ [⟨.result 24121 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 24121 .coefficient)
      LeftAuthority24120.bound (LeftAuthority24120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22132⟩⟩) (rawTerms := some (Proof.Events094.exact24121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24120.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24120.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority24120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound24129

namespace LeftBound24130
def owner : Owner := ⟨.program ⟨214⟩, ⟨22135⟩⟩
def transferEvent : Nat := 24130
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 24129) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 24129)
      LeftBound24129.bound (LeftBound24129.actual selector witness) := by
  exact .transfer (LeftBound24129.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound24129.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound24129.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound24129.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound24130

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
