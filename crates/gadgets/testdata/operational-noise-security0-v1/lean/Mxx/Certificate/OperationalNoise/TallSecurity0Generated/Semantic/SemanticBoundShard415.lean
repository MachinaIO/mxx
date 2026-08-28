import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard355
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard414

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61827
def owner : Owner := ⟨.program ⟨214⟩, ⟨29176⟩⟩
def transferEvent : Nat := 61827
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 52876 .summary) (.transfer 61826) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 52876 .summary)
      LeftBound52875.bound (LeftBound52875.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25457⟩⟩) (rawTerms := some (Proof.Events206.exact52876RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound52875.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61826)
      LeftBound61826.bound (LeftBound61826.actual selector witness) := by
  exact .transfer (LeftBound61826.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound52875.bound LeftBound61826.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound52875.bound, LeftBound61826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound52875.actual selector witness) * (LeftBound61826.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61827

namespace LeftBound61838
def owner : Owner := ⟨.program ⟨214⟩, ⟨22198⟩⟩
def transferEvent : Nat := 61838
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 61836 .coefficient) (.value (.predecessor 1 61837 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61836 .coefficient)
      LeftAuthority61834.bound (LeftAuthority61834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61835RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61834.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61837 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority61834.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61834.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61834.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound61838

namespace LeftBound61842
def owner : Owner := ⟨.program ⟨214⟩, ⟨22199⟩⟩
def transferEvent : Nat := 61842
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 61840 .coefficient) (.predecessor 1 61841 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61840 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61841 .coefficient)
      LeftBound61838.bound (LeftBound61838.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61839RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61838.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61838.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound61838.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound61838.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound61838.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61842

namespace LeftBound61843
def owner : Owner := ⟨.program ⟨214⟩, ⟨22199⟩⟩
def transferEvent : Nat := 61843
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22196⟩⟩]⟩ [⟨.result 61835 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61835 .coefficient)
      LeftAuthority61834.bound (LeftAuthority61834.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22196⟩⟩) (rawTerms := some (Proof.Events241.exact61835RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61834.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61834.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority61834.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority61834.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound61843

namespace LeftBound61844
def owner : Owner := ⟨.program ⟨214⟩, ⟨22199⟩⟩
def transferEvent : Nat := 61844
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 61843) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 61843)
      LeftBound61843.bound (LeftBound61843.actual selector witness) := by
  exact .transfer (LeftBound61843.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound61843.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound61843.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound61843.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61844

namespace LeftBound61939
def owner : Owner := ⟨.program ⟨214⟩, ⟨16554⟩⟩
def transferEvent : Nat := 61939
def frameStart : Nat := 61900
def rule : BoundRule := .identity (.predecessor 0 61938 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61938 .coefficient)
      LeftAuthority61936.bound (LeftAuthority61936.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61937RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61936.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61936.derived selector witness)

def rawBound : CoeffClass := LeftAuthority61936.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61936.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority61936.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61939

namespace LeftBound61956
def owner : Owner := ⟨.program ⟨214⟩, ⟨16593⟩⟩
def transferEvent : Nat := 61956
def frameStart : Nat := 61900
def rule : BoundRule := .sum [.predecessor 0 61954 .coefficient, .predecessor 1 61955 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61954 .coefficient)
      LeftBound61939.bound (LeftBound61939.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61955 .coefficient)
      LeftAuthority61952.bound (LeftAuthority61952.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority61952.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61939.bound, LeftAuthority61952.bound]
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61939.bound, LeftAuthority61952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61939.actual selector witness, LeftAuthority61952.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61956

namespace LeftBound61959
def owner : Owner := ⟨.program ⟨214⟩, ⟨16594⟩⟩
def transferEvent : Nat := 61959
def frameStart : Nat := 61900
def rule : BoundRule := .identity (.predecessor 0 61958 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61958 .coefficient)
      LeftBound61956.bound (LeftBound61956.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound61956.derived selector witness)

def rawBound : CoeffClass := LeftBound61956.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound61956.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound61959

namespace LeftBound61965
def owner : Owner := ⟨.program ⟨214⟩, ⟨16595⟩⟩
def transferEvent : Nat := 61965
def frameStart : Nat := 61900
def rule : BoundRule := .product (.predecessor 0 61963 .coefficient) (.predecessor 1 61964 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61963 .coefficient)
      LeftAuthority61961.bound (LeftAuthority61961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61964 .coefficient)
      LeftBound61959.bound (LeftBound61959.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61959.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61959.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority61961.bound LeftBound61959.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61961.bound, LeftBound61959.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority61961.actual selector witness) * (LeftBound61959.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61965

namespace LeftBound61973
def owner : Owner := ⟨.program ⟨214⟩, ⟨16596⟩⟩
def transferEvent : Nat := 61973
def frameStart : Nat := 61900
def rule : BoundRule := .sum [.predecessor 0 61971 .coefficient, .predecessor 1 61972 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61971 .coefficient)
      LeftAuthority61969.bound (LeftAuthority61969.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61970RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61969.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61969.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61972 .coefficient)
      LeftBound61965.bound (LeftBound61965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61965.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61965.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61969.bound, LeftBound61965.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61969.bound, LeftBound61965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61969.actual selector witness, LeftBound61965.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61973

namespace LeftBound61977
def owner : Owner := ⟨.program ⟨214⟩, ⟨29175⟩⟩
def transferEvent : Nat := 61977
def frameStart : Nat := 61900
def rule : BoundRule := .product (.predecessor 0 61975 .coefficient) (.predecessor 1 61976 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61975 .coefficient)
      LeftBound61973.bound (LeftBound61973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61973.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61976 .coefficient)
      LeftAuthority61950.bound (LeftAuthority61950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61950.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound61973.bound LeftAuthority61950.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61973.bound, LeftAuthority61950.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound61973.actual selector witness) * (LeftAuthority61950.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61977

namespace LeftBound61988
def owner : Owner := ⟨.program ⟨214⟩, ⟨17955⟩⟩
def transferEvent : Nat := 61988
def frameStart : Nat := 61900
def rule : BoundRule := .product (.predecessor 0 61986 .coefficient) (.predecessor 1 61987 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61986 .coefficient)
      LeftAuthority61961.bound (LeftAuthority61961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61962RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61961.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61987 .coefficient)
      LeftAuthority61984.bound (LeftAuthority61984.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61985RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61984.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61984.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority61961.bound LeftAuthority61984.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61961.bound, LeftAuthority61984.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority61961.actual selector witness) * (LeftAuthority61984.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound61988

namespace LeftBound61996
def owner : Owner := ⟨.program ⟨214⟩, ⟨17956⟩⟩
def transferEvent : Nat := 61996
def frameStart : Nat := 61900
def rule : BoundRule := .sum [.predecessor 0 61994 .coefficient, .predecessor 1 61995 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61994 .coefficient)
      LeftAuthority61992.bound (LeftAuthority61992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61993RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority61992.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority61992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61995 .coefficient)
      LeftBound61988.bound (LeftBound61988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61988.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority61992.bound, LeftBound61988.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority61992.bound, LeftBound61988.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority61992.actual selector witness, LeftBound61988.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61996

namespace LeftBound62000
def owner : Owner := ⟨.program ⟨214⟩, ⟨29180⟩⟩
def transferEvent : Nat := 62000
def frameStart : Nat := 61900
def rule : BoundRule := .sum [.predecessor 0 61998 .coefficient, .predecessor 1 61999 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 61998 .coefficient)
      LeftBound61996.bound (LeftBound61996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 61999 .coefficient)
      LeftBound61977.bound (LeftBound61977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact61982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61996.bound, LeftBound61977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61996.bound, LeftBound61977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61996.actual selector witness, LeftBound61977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62000

namespace LeftBound62013
def owner : Owner := ⟨.program ⟨214⟩, ⟨29177⟩⟩
def transferEvent : Nat := 62013
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62011 .coefficient, .predecessor 1 62012 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62011 .coefficient)
      LeftBound61842.bound (LeftBound61842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events242.exact62010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62012 .coefficient)
      LeftBound61825.bound (LeftBound61825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events241.exact61832RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61842.bound, LeftBound61825.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61842.bound, LeftBound61825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61842.actual selector witness, LeftBound61825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62013

namespace LeftBound62016
def owner : Owner := ⟨.program ⟨214⟩, ⟨29177⟩⟩
def transferEvent : Nat := 62016
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62010 .summary, .result 61832 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62010 .summary)
      LeftBound61844.bound (LeftBound61844.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22199⟩⟩) (rawTerms := some (Proof.Events242.exact62010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 61832 .summary)
      LeftBound61827.bound (LeftBound61827.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29176⟩⟩) (rawTerms := some (Proof.Events241.exact61832RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound61827.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61844.bound, LeftBound61827.bound]
def bound : CoeffClass := .finite ⟨1292337423279833362432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61844.bound, LeftBound61827.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound61844.actual selector witness, LeftBound61827.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62016

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
