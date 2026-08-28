import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard339
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50919
def owner : Owner := ⟨.program ⟨214⟩, ⟨17017⟩⟩
def transferEvent : Nat := 50919
def frameStart : Nat := 50817
def rule : BoundRule := .product (.predecessor 0 50917 .coefficient) (.predecessor 1 50918 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50917 .coefficient)
      LeftAuthority50872.bound (LeftAuthority50872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50918 .coefficient)
      LeftAuthority50915.bound (LeftAuthority50915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50915.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50915.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority50872.bound LeftAuthority50915.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50872.bound, LeftAuthority50915.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority50872.actual selector witness) * (LeftAuthority50915.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50919

namespace LeftBound50927
def owner : Owner := ⟨.program ⟨214⟩, ⟨17018⟩⟩
def transferEvent : Nat := 50927
def frameStart : Nat := 50817
def rule : BoundRule := .sum [.predecessor 0 50925 .coefficient, .predecessor 1 50926 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50925 .coefficient)
      LeftAuthority50923.bound (LeftAuthority50923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50926 .coefficient)
      LeftBound50919.bound (LeftBound50919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50919.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority50923.bound, LeftBound50919.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50923.bound, LeftBound50919.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority50923.actual selector witness, LeftBound50919.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50927

namespace LeftBound50931
def owner : Owner := ⟨.program ⟨214⟩, ⟨25767⟩⟩
def transferEvent : Nat := 50931
def frameStart : Nat := 50817
def rule : BoundRule := .sum [.predecessor 0 50929 .coefficient, .predecessor 1 50930 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50929 .coefficient)
      LeftBound50927.bound (LeftBound50927.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50927.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50930 .coefficient)
      LeftBound50908.bound (LeftBound50908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50913RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50908.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50927.bound, LeftBound50908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50927.bound, LeftBound50908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50927.actual selector witness, LeftBound50908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50931

namespace LeftBound50944
def owner : Owner := ⟨.program ⟨214⟩, ⟨25765⟩⟩
def transferEvent : Nat := 50944
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50942 .coefficient, .predecessor 1 50943 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50942 .coefficient)
      LeftBound50765.bound (LeftBound50765.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50765.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50765.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50943 .coefficient)
      LeftBound50737.bound (LeftBound50737.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50737.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50765.bound, LeftBound50737.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50765.bound, LeftBound50737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50765.actual selector witness, LeftBound50737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50944

namespace LeftBound50947
def owner : Owner := ⟨.program ⟨214⟩, ⟨25765⟩⟩
def transferEvent : Nat := 50947
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 50941 .summary, .result 50744 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50941 .summary)
      LeftBound50767.bound (LeftBound50767.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20255⟩⟩) (rawTerms := some (Proof.Events198.exact50941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50744 .summary)
      LeftBound50739.bound (LeftBound50739.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25764⟩⟩) (rawTerms := some (Proof.Events198.exact50744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50739.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50767.bound, LeftBound50739.bound]
def bound : CoeffClass := .finite ⟨352188964155392, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50767.bound, LeftBound50739.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50767.actual selector witness, LeftBound50739.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50947

namespace LeftBound50951
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def transferEvent : Nat := 50951
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50949 .coefficient) (.predecessor 1 50950 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50949 .coefficient)
      LeftBound50944.bound (LeftBound50944.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50944.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50944.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50950 .coefficient)
      LeftAuthority50654.bound (LeftAuthority50654.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50655RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50654.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50944.bound LeftAuthority50654.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50944.bound, LeftAuthority50654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50944.actual selector witness) * (LeftAuthority50654.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50951

namespace LeftBound50952
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def transferEvent : Nat := 50952
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30139⟩⟩]⟩ [⟨.result 50655 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50655 .coefficient)
      LeftAuthority50654.bound (LeftAuthority50654.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30139⟩⟩) (rawTerms := some (Proof.Events197.exact50655RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50654.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50654.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50654.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50654.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50952

namespace LeftBound50953
def owner : Owner := ⟨.program ⟨214⟩, ⟨30141⟩⟩
def transferEvent : Nat := 50953
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50948 .summary) (.transfer 50952) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50948 .summary)
      LeftBound50947.bound (LeftBound50947.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25765⟩⟩) (rawTerms := some (Proof.Events199.exact50948RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50952)
      LeftBound50952.bound (LeftBound50952.actual selector witness) := by
  exact .transfer (LeftBound50952.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50947.bound LeftBound50952.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50947.bound, LeftBound50952.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50947.actual selector witness) * (LeftBound50952.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50953

namespace LeftBound50964
def owner : Owner := ⟨.program ⟨214⟩, ⟨22846⟩⟩
def transferEvent : Nat := 50964
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 50962 .coefficient) (.value (.predecessor 1 50963 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50962 .coefficient)
      LeftAuthority50960.bound (LeftAuthority50960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50960.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50963 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority50960.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50960.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50960.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound50964

namespace LeftBound50968
def owner : Owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩
def transferEvent : Nat := 50968
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50966 .coefficient) (.predecessor 1 50967 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50966 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50967 .coefficient)
      LeftBound50964.bound (LeftBound50964.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact50965RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50964.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50964.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound50964.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound50964.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound50964.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50968

namespace LeftBound50969
def owner : Owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩
def transferEvent : Nat := 50969
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22844⟩⟩]⟩ [⟨.result 50961 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50961 .coefficient)
      LeftAuthority50960.bound (LeftAuthority50960.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22844⟩⟩) (rawTerms := some (Proof.Events199.exact50961RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50960.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50960.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50960.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50960.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50969

namespace LeftBound50970
def owner : Owner := ⟨.program ⟨214⟩, ⟨22847⟩⟩
def transferEvent : Nat := 50970
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 50969) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50969)
      LeftBound50969.bound (LeftBound50969.actual selector witness) := by
  exact .transfer (LeftBound50969.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound50969.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound50969.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound50969.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50970

namespace LeftBound51065
def owner : Owner := ⟨.program ⟨214⟩, ⟨17016⟩⟩
def transferEvent : Nat := 51065
def frameStart : Nat := 51026
def rule : BoundRule := .identity (.predecessor 0 51064 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51064 .coefficient)
      LeftAuthority51062.bound (LeftAuthority51062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51063RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51062.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51062.derived selector witness)

def rawBound : CoeffClass := LeftAuthority51062.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51062.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority51062.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51065

namespace LeftBound51082
def owner : Owner := ⟨.program ⟨214⟩, ⟨17055⟩⟩
def transferEvent : Nat := 51082
def frameStart : Nat := 51026
def rule : BoundRule := .sum [.predecessor 0 51080 .coefficient, .predecessor 1 51081 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51080 .coefficient)
      LeftBound51065.bound (LeftBound51065.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51065.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51081 .coefficient)
      LeftAuthority51078.bound (LeftAuthority51078.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority51078.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound51065.bound, LeftAuthority51078.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51065.bound, LeftAuthority51078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound51065.actual selector witness, LeftAuthority51078.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound51082

namespace LeftBound51085
def owner : Owner := ⟨.program ⟨214⟩, ⟨17056⟩⟩
def transferEvent : Nat := 51085
def frameStart : Nat := 51026
def rule : BoundRule := .identity (.predecessor 0 51084 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51084 .coefficient)
      LeftBound51082.bound (LeftBound51082.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound51082.derived selector witness)

def rawBound : CoeffClass := LeftBound51082.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound51082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound51082.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound51085

namespace LeftBound51091
def owner : Owner := ⟨.program ⟨214⟩, ⟨17057⟩⟩
def transferEvent : Nat := 51091
def frameStart : Nat := 51026
def rule : BoundRule := .product (.predecessor 0 51089 .coefficient) (.predecessor 1 51090 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 51089 .coefficient)
      LeftAuthority51087.bound (LeftAuthority51087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority51087.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority51087.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 51090 .coefficient)
      LeftBound51085.bound (LeftBound51085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events199.exact51086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound51085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound51085.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority51087.bound LeftBound51085.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority51087.bound, LeftBound51085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority51087.actual selector witness) * (LeftBound51085.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound51091

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
