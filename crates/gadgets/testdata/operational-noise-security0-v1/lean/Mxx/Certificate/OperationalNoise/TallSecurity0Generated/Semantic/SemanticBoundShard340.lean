import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard339

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound50754
def owner : Owner := ⟨.program ⟨214⟩, ⟨5546⟩⟩
def transferEvent : Nat := 50754
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50752 .coefficient) (.predecessor 1 50753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50752 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50753 .coefficient)
      LeftAuthority6549.bound (LeftAuthority6549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6549.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftAuthority6549.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftAuthority6549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftAuthority6549.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 16) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50754

namespace LeftBound50759
def owner : Owner := ⟨.program ⟨214⟩, ⟨5547⟩⟩
def transferEvent : Nat := 50759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 50757 .coefficient, .predecessor 1 50758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50757 .coefficient)
      LeftBound50754.bound (LeftBound50754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50758 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50754.bound, LeftAuthority6547.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50754.bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50754.actual selector witness, LeftAuthority6547.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50759

namespace LeftBound50760
def owner : Owner := ⟨.program ⟨214⟩, ⟨5547⟩⟩
def transferEvent : Nat := 50760
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22⟩⟩]⟩ [⟨.result 6548 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6548 .coefficient)
      LeftAuthority6547.bound (LeftAuthority6547.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22⟩⟩) (rawTerms := some (Proof.Events025.exact6548RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6547.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6547.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6547.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6547.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6547.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50760

namespace LeftBound50765
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def transferEvent : Nat := 50765
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 50763 .coefficient) (.predecessor 1 50764 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50763 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50764 .coefficient)
      LeftBound50750.bound (LeftBound50750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50750.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50750.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound50750.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound50750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound50750.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50765

namespace LeftBound50766
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def transferEvent : Nat := 50766
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20252⟩⟩]⟩ [⟨.result 50747 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50747 .coefficient)
      LeftAuthority50746.bound (LeftAuthority50746.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20252⟩⟩) (rawTerms := some (Proof.Events198.exact50747RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50746.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50746.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority50746.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50746.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50746.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound50766

namespace LeftBound50767
def owner : Owner := ⟨.program ⟨214⟩, ⟨20255⟩⟩
def transferEvent : Nat := 50767
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 50766) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 50766)
      LeftBound50766.bound (LeftBound50766.actual selector witness) := by
  exact .transfer (LeftBound50766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound50766.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound50766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound50766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50767

namespace LeftBound50846
def owner : Owner := ⟨.program ⟨214⟩, ⟨13359⟩⟩
def transferEvent : Nat := 50846
def frameStart : Nat := 50817
def rule : BoundRule := .product (.predecessor 0 50844 .coefficient) (.predecessor 1 50845 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50844 .coefficient)
      LeftAuthority50842.bound (LeftAuthority50842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50843RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50842.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50842.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50845 .coefficient)
      LeftAuthority50839.bound (LeftAuthority50839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50840RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50839.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50839.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority50842.bound LeftAuthority50839.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50842.bound, LeftAuthority50839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority50842.actual selector witness) * (LeftAuthority50839.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50846

namespace LeftBound50850
def owner : Owner := ⟨.program ⟨214⟩, ⟨13360⟩⟩
def transferEvent : Nat := 50850
def frameStart : Nat := 50817
def rule : BoundRule := .identity (.predecessor 0 50849 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50849 .coefficient)
      LeftBound50846.bound (LeftBound50846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50848RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50846.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50846.derived selector witness)

def rawBound : CoeffClass := LeftBound50846.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound50846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50850

namespace LeftBound50867
def owner : Owner := ⟨.program ⟨214⟩, ⟨13450⟩⟩
def transferEvent : Nat := 50867
def frameStart : Nat := 50817
def rule : BoundRule := .sum [.predecessor 0 50865 .coefficient, .predecessor 1 50866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50865 .coefficient)
      LeftBound50850.bound (LeftBound50850.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound50850.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50866 .coefficient)
      LeftAuthority50863.bound (LeftAuthority50863.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority50863.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50850.bound, LeftAuthority50863.bound]
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50850.bound, LeftAuthority50863.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50850.actual selector witness, LeftAuthority50863.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50867

namespace LeftBound50870
def owner : Owner := ⟨.program ⟨214⟩, ⟨13451⟩⟩
def transferEvent : Nat := 50870
def frameStart : Nat := 50817
def rule : BoundRule := .identity (.predecessor 0 50869 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50869 .coefficient)
      LeftBound50867.bound (LeftBound50867.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound50867.derived selector witness)

def rawBound : CoeffClass := LeftBound50867.bound
def bound : CoeffClass := .finite ⟨3600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50867.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound50867.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50870

namespace LeftBound50876
def owner : Owner := ⟨.program ⟨214⟩, ⟨13452⟩⟩
def transferEvent : Nat := 50876
def frameStart : Nat := 50817
def rule : BoundRule := .product (.predecessor 0 50874 .coefficient) (.predecessor 1 50875 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50874 .coefficient)
      LeftAuthority50872.bound (LeftAuthority50872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50873RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50872.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50875 .coefficient)
      LeftBound50870.bound (LeftBound50870.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50871RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50870.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50870.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority50872.bound LeftBound50870.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50872.bound, LeftBound50870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority50872.actual selector witness) * (LeftBound50870.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50876

namespace LeftBound50892
def owner : Owner := ⟨.program ⟨214⟩, ⟨7883⟩⟩
def transferEvent : Nat := 50892
def frameStart : Nat := 50817
def rule : BoundRule := .scale (.predecessor 0 50890 .coefficient) (.value (.predecessor 1 50891 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50890 .coefficient)
      LeftAuthority50888.bound (LeftAuthority50888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50891 .coefficient)
      LeftAuthority50879.bound (LeftAuthority50879.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority50879.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority50888.bound LeftAuthority50879.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50888.bound, LeftAuthority50879.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority50888.actual selector witness) * (LeftAuthority50879.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound50892

namespace LeftBound50895
def owner : Owner := ⟨.program ⟨214⟩, ⟨6770⟩⟩
def transferEvent : Nat := 50895
def frameStart : Nat := 50817
def rule : BoundRule := .identity (.predecessor 0 50894 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50894 .coefficient)
      LeftAuthority50882.bound (LeftAuthority50882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50883RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50882.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50882.derived selector witness)

def rawBound : CoeffClass := LeftAuthority50882.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority50882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority50882.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound50895

namespace LeftBound50899
def owner : Owner := ⟨.program ⟨214⟩, ⟨7884⟩⟩
def transferEvent : Nat := 50899
def frameStart : Nat := 50817
def rule : BoundRule := .product (.predecessor 0 50897 .coefficient) (.predecessor 1 50898 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50897 .coefficient)
      LeftBound50895.bound (LeftBound50895.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50896RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50895.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50895.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50898 .coefficient)
      LeftBound50892.bound (LeftBound50892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50892.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50895.bound LeftBound50892.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50895.bound, LeftBound50892.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50895.actual selector witness) * (LeftBound50892.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50899

namespace LeftBound50904
def owner : Owner := ⟨.program ⟨214⟩, ⟨13453⟩⟩
def transferEvent : Nat := 50904
def frameStart : Nat := 50817
def rule : BoundRule := .sum [.predecessor 0 50902 .coefficient, .predecessor 1 50903 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50902 .coefficient)
      LeftBound50899.bound (LeftBound50899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50901RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50899.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50903 .coefficient)
      LeftBound50876.bound (LeftBound50876.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50878RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50876.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50876.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound50899.bound, LeftBound50876.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50899.bound, LeftBound50876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound50899.actual selector witness, LeftBound50876.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound50904

namespace LeftBound50908
def owner : Owner := ⟨.program ⟨214⟩, ⟨25766⟩⟩
def transferEvent : Nat := 50908
def frameStart : Nat := 50817
def rule : BoundRule := .product (.predecessor 0 50906 .coefficient) (.predecessor 1 50907 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 50906 .coefficient)
      LeftBound50904.bound (LeftBound50904.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50904.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50904.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 50907 .coefficient)
      LeftAuthority50861.bound (LeftAuthority50861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority50861.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority50861.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound50904.bound LeftAuthority50861.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50904.bound, LeftAuthority50861.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound50904.actual selector witness) * (LeftAuthority50861.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound50908

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
