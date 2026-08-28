import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard445

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66015
def owner : Owner := ⟨.program ⟨214⟩, ⟨25679⟩⟩
def transferEvent : Nat := 66015
def frameStart : Nat := 65924
def rule : BoundRule := .product (.predecessor 0 66013 .coefficient) (.predecessor 1 66014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66013 .coefficient)
      LeftBound66011.bound (LeftBound66011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66014 .coefficient)
      LeftAuthority65968.bound (LeftAuthority65968.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65968.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65968.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66011.bound LeftAuthority65968.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66011.bound, LeftAuthority65968.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66011.actual selector witness) * (LeftAuthority65968.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66015

namespace LeftBound66026
def owner : Owner := ⟨.program ⟨214⟩, ⟨16869⟩⟩
def transferEvent : Nat := 66026
def frameStart : Nat := 65924
def rule : BoundRule := .product (.predecessor 0 66024 .coefficient) (.predecessor 1 66025 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66024 .coefficient)
      LeftAuthority65979.bound (LeftAuthority65979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65980RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65979.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66025 .coefficient)
      LeftAuthority66022.bound (LeftAuthority66022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority65979.bound LeftAuthority66022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65979.bound, LeftAuthority66022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority65979.actual selector witness) * (LeftAuthority66022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66026

namespace LeftBound66034
def owner : Owner := ⟨.program ⟨214⟩, ⟨16870⟩⟩
def transferEvent : Nat := 66034
def frameStart : Nat := 65924
def rule : BoundRule := .sum [.predecessor 0 66032 .coefficient, .predecessor 1 66033 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66032 .coefficient)
      LeftAuthority66030.bound (LeftAuthority66030.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66031RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66030.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66030.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66033 .coefficient)
      LeftBound66026.bound (LeftBound66026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66026.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority66030.bound, LeftBound66026.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66030.bound, LeftBound66026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority66030.actual selector witness, LeftBound66026.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66034

namespace LeftBound66038
def owner : Owner := ⟨.program ⟨214⟩, ⟨25680⟩⟩
def transferEvent : Nat := 66038
def frameStart : Nat := 65924
def rule : BoundRule := .sum [.predecessor 0 66036 .coefficient, .predecessor 1 66037 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66036 .coefficient)
      LeftBound66034.bound (LeftBound66034.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66034.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66037 .coefficient)
      LeftBound66015.bound (LeftBound66015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact66020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66034.bound, LeftBound66015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66034.bound, LeftBound66015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66034.actual selector witness, LeftBound66015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66038

namespace LeftBound66051
def owner : Owner := ⟨.program ⟨214⟩, ⟨25678⟩⟩
def transferEvent : Nat := 66051
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66049 .coefficient, .predecessor 1 66050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66049 .coefficient)
      LeftBound65872.bound (LeftBound65872.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65872.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65872.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66050 .coefficient)
      LeftBound65855.bound (LeftBound65855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events257.exact65862RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65855.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65872.bound, LeftBound65855.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65872.bound, LeftBound65855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65872.actual selector witness, LeftBound65855.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66051

namespace LeftBound66054
def owner : Owner := ⟨.program ⟨214⟩, ⟨25678⟩⟩
def transferEvent : Nat := 66054
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66048 .summary, .result 65862 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66048 .summary)
      LeftBound65874.bound (LeftBound65874.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20175⟩⟩) (rawTerms := some (Proof.Events258.exact66048RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65874.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65862 .summary)
      LeftBound65857.bound (LeftBound65857.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25677⟩⟩) (rawTerms := some (Proof.Events257.exact65862RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65857.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65874.bound, LeftBound65857.bound]
def bound : CoeffClass := .finite ⟨352182857248768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65874.bound, LeftBound65857.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound65874.actual selector witness, LeftBound65857.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66054

namespace LeftBound66058
def owner : Owner := ⟨.program ⟨214⟩, ⟨29808⟩⟩
def transferEvent : Nat := 66058
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66056 .coefficient) (.predecessor 1 66057 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66056 .coefficient)
      LeftBound66051.bound (LeftBound66051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66055RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66057 .coefficient)
      LeftAuthority65777.bound (LeftAuthority65777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events256.exact65778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65777.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66051.bound LeftAuthority65777.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66051.bound, LeftAuthority65777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66051.actual selector witness) * (LeftAuthority65777.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66058

namespace LeftBound66059
def owner : Owner := ⟨.program ⟨214⟩, ⟨29808⟩⟩
def transferEvent : Nat := 66059
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29806⟩⟩]⟩ [⟨.result 65778 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65778 .coefficient)
      LeftAuthority65777.bound (LeftAuthority65777.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29806⟩⟩) (rawTerms := some (Proof.Events256.exact65778RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority65777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority65777.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority65777.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority65777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority65777.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66059

namespace LeftBound66060
def owner : Owner := ⟨.program ⟨214⟩, ⟨29808⟩⟩
def transferEvent : Nat := 66060
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66055 .summary) (.transfer 66059) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66055 .summary)
      LeftBound66054.bound (LeftBound66054.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25678⟩⟩) (rawTerms := some (Proof.Events258.exact66055RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66054.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66059)
      LeftBound66059.bound (LeftBound66059.actual selector witness) := by
  exact .transfer (LeftBound66059.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66054.bound LeftBound66059.bound
def bound : CoeffClass := .finite ⟨1292516721028694540288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66054.bound, LeftBound66059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66054.actual selector witness) * (LeftBound66059.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66060

namespace LeftBound66071
def owner : Owner := ⟨.program ⟨214⟩, ⟨22694⟩⟩
def transferEvent : Nat := 66071
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 66069 .coefficient) (.value (.predecessor 1 66070 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66069 .coefficient)
      LeftAuthority66067.bound (LeftAuthority66067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66070 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66067.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66067.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66067.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66071

namespace LeftBound66075
def owner : Owner := ⟨.program ⟨214⟩, ⟨22695⟩⟩
def transferEvent : Nat := 66075
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66073 .coefficient) (.predecessor 1 66074 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66073 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66074 .coefficient)
      LeftBound66071.bound (LeftBound66071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66071.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound66071.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound66071.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound66071.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66075

namespace LeftBound66076
def owner : Owner := ⟨.program ⟨214⟩, ⟨22695⟩⟩
def transferEvent : Nat := 66076
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22692⟩⟩]⟩ [⟨.result 66068 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66068 .coefficient)
      LeftAuthority66067.bound (LeftAuthority66067.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22692⟩⟩) (rawTerms := some (Proof.Events258.exact66068RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66067.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66067.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66067.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66067.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66076

namespace LeftBound66077
def owner : Owner := ⟨.program ⟨214⟩, ⟨22695⟩⟩
def transferEvent : Nat := 66077
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 66076) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66076)
      LeftBound66076.bound (LeftBound66076.actual selector witness) := by
  exact .transfer (LeftBound66076.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound66076.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound66076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound66076.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66077

namespace LeftBound66172
def owner : Owner := ⟨.program ⟨214⟩, ⟨16868⟩⟩
def transferEvent : Nat := 66172
def frameStart : Nat := 66133
def rule : BoundRule := .identity (.predecessor 0 66171 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66171 .coefficient)
      LeftAuthority66169.bound (LeftAuthority66169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66170RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66169.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66169.derived selector witness)

def rawBound : CoeffClass := LeftAuthority66169.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority66169.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66172

namespace LeftBound66189
def owner : Owner := ⟨.program ⟨214⟩, ⟨16963⟩⟩
def transferEvent : Nat := 66189
def frameStart : Nat := 66133
def rule : BoundRule := .sum [.predecessor 0 66187 .coefficient, .predecessor 1 66188 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66187 .coefficient)
      LeftBound66172.bound (LeftBound66172.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66172.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66188 .coefficient)
      LeftAuthority66185.bound (LeftAuthority66185.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66172.bound, LeftAuthority66185.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66172.bound, LeftAuthority66185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66172.actual selector witness, LeftAuthority66185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66189

namespace LeftBound66192
def owner : Owner := ⟨.program ⟨214⟩, ⟨16964⟩⟩
def transferEvent : Nat := 66192
def frameStart : Nat := 66133
def rule : BoundRule := .identity (.predecessor 0 66191 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66191 .coefficient)
      LeftBound66189.bound (LeftBound66189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66189.derived selector witness)

def rawBound : CoeffClass := LeftBound66189.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66189.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66192

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
