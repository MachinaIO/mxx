import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard695

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound100976
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def transferEvent : Nat := 100976
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19157⟩⟩]⟩ [⟨.result 100968 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100968 .coefficient)
      LeftAuthority100967.bound (LeftAuthority100967.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19157⟩⟩) (rawTerms := some (Proof.Events394.exact100968RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100967.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100967.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100967.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100967.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound100976

namespace LeftBound100977
def owner : Owner := ⟨.program ⟨214⟩, ⟨19160⟩⟩
def transferEvent : Nat := 100977
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 100976) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 100976)
      LeftBound100976.bound (LeftBound100976.actual selector witness) := by
  exact .transfer (LeftBound100976.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound100976.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound100976.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound100976.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound100977

namespace LeftBound101032
def owner : Owner := ⟨.program ⟨214⟩, ⟨10954⟩⟩
def transferEvent : Nat := 101032
def frameStart : Nat := 101015
def rule : BoundRule := .product (.predecessor 0 101030 .coefficient) (.predecessor 1 101031 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101030 .coefficient)
      LeftAuthority101028.bound (LeftAuthority101028.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101029RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101028.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101028.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101031 .coefficient)
      LeftAuthority101025.bound (LeftAuthority101025.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101026RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101025.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101028.bound LeftAuthority101025.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101028.bound, LeftAuthority101025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101028.actual selector witness) * (LeftAuthority101025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101032

namespace LeftBound101036
def owner : Owner := ⟨.program ⟨214⟩, ⟨10955⟩⟩
def transferEvent : Nat := 101036
def frameStart : Nat := 101015
def rule : BoundRule := .identity (.predecessor 0 101035 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101035 .coefficient)
      LeftBound101032.bound (LeftBound101032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101034RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101032.derived selector witness)

def rawBound : CoeffClass := LeftBound101032.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101032.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101032.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101036

namespace LeftBound101053
def owner : Owner := ⟨.program ⟨214⟩, ⟨11065⟩⟩
def transferEvent : Nat := 101053
def frameStart : Nat := 101015
def rule : BoundRule := .sum [.predecessor 0 101051 .coefficient, .predecessor 1 101052 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101051 .coefficient)
      LeftBound101036.bound (LeftBound101036.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101052 .coefficient)
      LeftAuthority101049.bound (LeftAuthority101049.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101049.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101036.bound, LeftAuthority101049.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101036.bound, LeftAuthority101049.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101036.actual selector witness, LeftAuthority101049.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101053

namespace LeftBound101056
def owner : Owner := ⟨.program ⟨214⟩, ⟨11066⟩⟩
def transferEvent : Nat := 101056
def frameStart : Nat := 101015
def rule : BoundRule := .identity (.predecessor 0 101055 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101055 .coefficient)
      LeftBound101053.bound (LeftBound101053.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101053.derived selector witness)

def rawBound : CoeffClass := LeftBound101053.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101053.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101053.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101056

namespace LeftBound101062
def owner : Owner := ⟨.program ⟨214⟩, ⟨11067⟩⟩
def transferEvent : Nat := 101062
def frameStart : Nat := 101015
def rule : BoundRule := .product (.predecessor 0 101060 .coefficient) (.predecessor 1 101061 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101060 .coefficient)
      LeftAuthority101058.bound (LeftAuthority101058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101061 .coefficient)
      LeftBound101056.bound (LeftBound101056.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101057RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101056.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101056.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority101058.bound LeftBound101056.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101058.bound, LeftBound101056.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority101058.actual selector witness) * (LeftBound101056.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101062

namespace LeftBound101078
def owner : Owner := ⟨.program ⟨214⟩, ⟨7838⟩⟩
def transferEvent : Nat := 101078
def frameStart : Nat := 101015
def rule : BoundRule := .scale (.predecessor 0 101076 .coefficient) (.value (.predecessor 1 101077 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101076 .coefficient)
      LeftAuthority101074.bound (LeftAuthority101074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101074.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101077 .coefficient)
      LeftAuthority101065.bound (LeftAuthority101065.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101065.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101074.bound LeftAuthority101065.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101074.bound, LeftAuthority101065.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101074.actual selector witness) * (LeftAuthority101065.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101078

namespace LeftBound101081
def owner : Owner := ⟨.program ⟨214⟩, ⟨6791⟩⟩
def transferEvent : Nat := 101081
def frameStart : Nat := 101015
def rule : BoundRule := .identity (.predecessor 0 101080 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101080 .coefficient)
      LeftAuthority101068.bound (LeftAuthority101068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101068.derived selector witness)

def rawBound : CoeffClass := LeftAuthority101068.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority101068.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101081

namespace LeftBound101085
def owner : Owner := ⟨.program ⟨214⟩, ⟨7839⟩⟩
def transferEvent : Nat := 101085
def frameStart : Nat := 101015
def rule : BoundRule := .product (.predecessor 0 101083 .coefficient) (.predecessor 1 101084 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101083 .coefficient)
      LeftBound101081.bound (LeftBound101081.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101081.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101081.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101084 .coefficient)
      LeftBound101078.bound (LeftBound101078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101078.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101078.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101081.bound LeftBound101078.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101081.bound, LeftBound101078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101081.actual selector witness) * (LeftBound101078.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101085

namespace LeftBound101090
def owner : Owner := ⟨.program ⟨214⟩, ⟨11068⟩⟩
def transferEvent : Nat := 101090
def frameStart : Nat := 101015
def rule : BoundRule := .sum [.predecessor 0 101088 .coefficient, .predecessor 1 101089 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101088 .coefficient)
      LeftBound101085.bound (LeftBound101085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101089 .coefficient)
      LeftBound101062.bound (LeftBound101062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101062.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101085.bound, LeftBound101062.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101085.bound, LeftBound101062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101085.actual selector witness, LeftBound101062.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101090

namespace LeftBound101094
def owner : Owner := ⟨.program ⟨214⟩, ⟨25055⟩⟩
def transferEvent : Nat := 101094
def frameStart : Nat := 101015
def rule : BoundRule := .product (.predecessor 0 101092 .coefficient) (.predecessor 1 101093 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101092 .coefficient)
      LeftBound101090.bound (LeftBound101090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101093 .coefficient)
      LeftAuthority101047.bound (LeftAuthority101047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101047.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101047.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101090.bound LeftAuthority101047.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101090.bound, LeftAuthority101047.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101090.actual selector witness) * (LeftAuthority101047.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101094

namespace LeftBound101105
def owner : Owner := ⟨.program ⟨214⟩, ⟨15106⟩⟩
def transferEvent : Nat := 101105
def frameStart : Nat := 101015
def rule : BoundRule := .product (.predecessor 0 101103 .coefficient) (.predecessor 1 101104 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101103 .coefficient)
      LeftAuthority101058.bound (LeftAuthority101058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101058.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101104 .coefficient)
      LeftAuthority101101.bound (LeftAuthority101101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101101.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101101.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101058.bound LeftAuthority101101.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101058.bound, LeftAuthority101101.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101058.actual selector witness) * (LeftAuthority101101.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101105

namespace LeftBound101113
def owner : Owner := ⟨.program ⟨214⟩, ⟨15107⟩⟩
def transferEvent : Nat := 101113
def frameStart : Nat := 101015
def rule : BoundRule := .sum [.predecessor 0 101111 .coefficient, .predecessor 1 101112 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101111 .coefficient)
      LeftAuthority101109.bound (LeftAuthority101109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101109.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101112 .coefficient)
      LeftBound101105.bound (LeftBound101105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101107RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101109.bound, LeftBound101105.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101109.bound, LeftBound101105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101109.actual selector witness, LeftBound101105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101113

namespace LeftBound101117
def owner : Owner := ⟨.program ⟨214⟩, ⟨25056⟩⟩
def transferEvent : Nat := 101117
def frameStart : Nat := 101015
def rule : BoundRule := .sum [.predecessor 0 101115 .coefficient, .predecessor 1 101116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101115 .coefficient)
      LeftBound101113.bound (LeftBound101113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101116 .coefficient)
      LeftBound101094.bound (LeftBound101094.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact101099RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101094.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101094.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101113.bound, LeftBound101094.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101113.bound, LeftBound101094.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101113.actual selector witness, LeftBound101094.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101117

namespace LeftBound101130
def owner : Owner := ⟨.program ⟨214⟩, ⟨25054⟩⟩
def transferEvent : Nat := 101130
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 101128 .coefficient, .predecessor 1 101129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101128 .coefficient)
      LeftBound100975.bound (LeftBound100975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101129 .coefficient)
      LeftBound100958.bound (LeftBound100958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound100958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound100958.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100975.bound, LeftBound100958.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100975.bound, LeftBound100958.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100975.actual selector witness, LeftBound100958.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101130

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
