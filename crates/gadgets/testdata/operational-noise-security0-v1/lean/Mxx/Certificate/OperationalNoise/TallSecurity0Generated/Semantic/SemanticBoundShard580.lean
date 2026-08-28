import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard579

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84952
def owner : Owner := ⟨.program ⟨214⟩, ⟨14317⟩⟩
def transferEvent : Nat := 84952
def frameStart : Nat := 84867
def rule : BoundRule := .sum [.predecessor 0 84950 .coefficient, .predecessor 1 84951 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84950 .coefficient)
      LeftBound84947.bound (LeftBound84947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84947.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84951 .coefficient)
      LeftBound84926.bound (LeftBound84926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84928RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84947.bound, LeftBound84926.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84947.bound, LeftBound84926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84947.actual selector witness, LeftBound84926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84952

namespace LeftBound84956
def owner : Owner := ⟨.program ⟨214⟩, ⟨26069⟩⟩
def transferEvent : Nat := 84956
def frameStart : Nat := 84867
def rule : BoundRule := .product (.predecessor 0 84954 .coefficient) (.predecessor 1 84955 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84954 .coefficient)
      LeftBound84952.bound (LeftBound84952.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84953RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84952.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84952.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84955 .coefficient)
      LeftAuthority84911.bound (LeftAuthority84911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84911.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84952.bound LeftAuthority84911.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84952.bound, LeftAuthority84911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84952.actual selector witness) * (LeftAuthority84911.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84956

namespace LeftBound84967
def owner : Owner := ⟨.program ⟨214⟩, ⟨15942⟩⟩
def transferEvent : Nat := 84967
def frameStart : Nat := 84867
def rule : BoundRule := .product (.predecessor 0 84965 .coefficient) (.predecessor 1 84966 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84965 .coefficient)
      LeftAuthority84922.bound (LeftAuthority84922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84922.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84966 .coefficient)
      LeftAuthority84963.bound (LeftAuthority84963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84922.bound LeftAuthority84963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84922.bound, LeftAuthority84963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84922.actual selector witness) * (LeftAuthority84963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84967

namespace LeftBound84975
def owner : Owner := ⟨.program ⟨214⟩, ⟨15943⟩⟩
def transferEvent : Nat := 84975
def frameStart : Nat := 84867
def rule : BoundRule := .sum [.predecessor 0 84973 .coefficient, .predecessor 1 84974 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84973 .coefficient)
      LeftAuthority84971.bound (LeftAuthority84971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84972RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84971.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84971.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84974 .coefficient)
      LeftBound84967.bound (LeftBound84967.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84969RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84967.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84967.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84971.bound, LeftBound84967.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84971.bound, LeftBound84967.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84971.actual selector witness, LeftBound84967.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84975

namespace LeftBound84979
def owner : Owner := ⟨.program ⟨214⟩, ⟨26070⟩⟩
def transferEvent : Nat := 84979
def frameStart : Nat := 84867
def rule : BoundRule := .sum [.predecessor 0 84977 .coefficient, .predecessor 1 84978 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84977 .coefficient)
      LeftBound84975.bound (LeftBound84975.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84976RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84975.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84975.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84978 .coefficient)
      LeftBound84956.bound (LeftBound84956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84956.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84956.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84975.bound, LeftBound84956.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84975.bound, LeftBound84956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84975.actual selector witness, LeftBound84956.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84979

namespace LeftBound84992
def owner : Owner := ⟨.program ⟨214⟩, ⟨26068⟩⟩
def transferEvent : Nat := 84992
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84990 .coefficient, .predecessor 1 84991 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84990 .coefficient)
      LeftBound84815.bound (LeftBound84815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84989RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84815.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84991 .coefficient)
      LeftBound84798.bound (LeftBound84798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events331.exact84805RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84815.bound, LeftBound84798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84815.bound, LeftBound84798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84815.actual selector witness, LeftBound84798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84992

namespace LeftBound84995
def owner : Owner := ⟨.program ⟨214⟩, ⟨26068⟩⟩
def transferEvent : Nat := 84995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84989 .summary, .result 84805 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84989 .summary)
      LeftBound84817.bound (LeftBound84817.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19531⟩⟩) (rawTerms := some (Proof.Events331.exact84989RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84817.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84805 .summary)
      LeftBound84800.bound (LeftBound84800.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26067⟩⟩) (rawTerms := some (Proof.Events331.exact84805RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84817.bound, LeftBound84800.bound]
def bound : CoeffClass := .finite ⟨352060719116288, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84817.bound, LeftBound84800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84817.actual selector witness, LeftBound84800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84995

namespace LeftBound84999
def owner : Owner := ⟨.program ⟨214⟩, ⟨27868⟩⟩
def transferEvent : Nat := 84999
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84997 .coefficient) (.predecessor 1 84998 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84997 .coefficient)
      LeftBound84992.bound (LeftBound84992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact84996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84998 .coefficient)
      LeftAuthority84720.bound (LeftAuthority84720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84720.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84992.bound LeftAuthority84720.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84992.bound, LeftAuthority84720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84992.actual selector witness) * (LeftAuthority84720.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84999

namespace LeftBound85000
def owner : Owner := ⟨.program ⟨214⟩, ⟨27868⟩⟩
def transferEvent : Nat := 85000
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27866⟩⟩]⟩ [⟨.result 84721 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84721 .coefficient)
      LeftAuthority84720.bound (LeftAuthority84720.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27866⟩⟩) (rawTerms := some (Proof.Events330.exact84721RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84720.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84720.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84720.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85000

namespace LeftBound85001
def owner : Owner := ⟨.program ⟨214⟩, ⟨27868⟩⟩
def transferEvent : Nat := 85001
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 84996 .summary) (.transfer 85000) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84996 .summary)
      LeftBound84995.bound (LeftBound84995.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26068⟩⟩) (rawTerms := some (Proof.Events332.exact84996RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85000)
      LeftBound85000.bound (LeftBound85000.actual selector witness) := by
  exact .transfer (LeftBound85000.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84995.bound LeftBound85000.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84995.bound, LeftBound85000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84995.actual selector witness) * (LeftBound85000.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85001

namespace LeftBound85012
def owner : Owner := ⟨.program ⟨214⟩, ⟨21402⟩⟩
def transferEvent : Nat := 85012
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 85010 .coefficient) (.value (.predecessor 1 85011 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85010 .coefficient)
      LeftAuthority85008.bound (LeftAuthority85008.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85009RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85008.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85011 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority85008.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85008.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85008.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound85012

namespace LeftBound85016
def owner : Owner := ⟨.program ⟨214⟩, ⟨21403⟩⟩
def transferEvent : Nat := 85016
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 85014 .coefficient) (.predecessor 1 85015 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85014 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85015 .coefficient)
      LeftBound85012.bound (LeftBound85012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85012.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound85012.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound85012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound85012.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85016

namespace LeftBound85017
def owner : Owner := ⟨.program ⟨214⟩, ⟨21403⟩⟩
def transferEvent : Nat := 85017
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21400⟩⟩]⟩ [⟨.result 85009 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85009 .coefficient)
      LeftAuthority85008.bound (LeftAuthority85008.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21400⟩⟩) (rawTerms := some (Proof.Events332.exact85009RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85008.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85008.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority85008.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority85008.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound85017

namespace LeftBound85018
def owner : Owner := ⟨.program ⟨214⟩, ⟨21403⟩⟩
def transferEvent : Nat := 85018
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 85017) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 85017)
      LeftBound85017.bound (LeftBound85017.actual selector witness) := by
  exact .transfer (LeftBound85017.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound85017.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound85017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound85017.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound85018

namespace LeftBound85113
def owner : Owner := ⟨.program ⟨214⟩, ⟨15941⟩⟩
def transferEvent : Nat := 85113
def frameStart : Nat := 85074
def rule : BoundRule := .identity (.predecessor 0 85112 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85112 .coefficient)
      LeftAuthority85110.bound (LeftAuthority85110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events332.exact85111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85110.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85110.derived selector witness)

def rawBound : CoeffClass := LeftAuthority85110.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority85110.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound85113

namespace LeftBound85130
def owner : Owner := ⟨.program ⟨214⟩, ⟨16015⟩⟩
def transferEvent : Nat := 85130
def frameStart : Nat := 85074
def rule : BoundRule := .sum [.predecessor 0 85128 .coefficient, .predecessor 1 85129 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 85128 .coefficient)
      LeftBound85113.bound (LeftBound85113.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound85113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 85129 .coefficient)
      LeftAuthority85126.bound (LeftAuthority85126.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority85126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85113.bound, LeftAuthority85126.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85113.bound, LeftAuthority85126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound85113.actual selector witness, LeftAuthority85126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85130

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
