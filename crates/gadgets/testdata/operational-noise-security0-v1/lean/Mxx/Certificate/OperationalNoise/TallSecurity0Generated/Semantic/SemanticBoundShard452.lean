import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard451

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66804
def owner : Owner := ⟨.program ⟨214⟩, ⟨10029⟩⟩
def transferEvent : Nat := 66804
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7873⟩⟩]⟩ [⟨.result 8001 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8001 .coefficient)
      LeftAuthority8000.bound (LeftAuthority8000.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7873⟩⟩) (rawTerms := some (Proof.Events031.exact8001RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8000.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8000.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8000.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8000.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8000.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66804

namespace LeftBound66805
def owner : Owner := ⟨.program ⟨214⟩, ⟨10029⟩⟩
def transferEvent : Nat := 66805
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66800 .summary) (.transfer 66804) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66800 .summary)
      LeftBound66798.bound (LeftBound66798.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10028⟩⟩) (rawTerms := some (Proof.Events260.exact66800RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66798.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66804)
      LeftBound66804.bound (LeftBound66804.actual selector witness) := by
  exact .transfer (LeftBound66804.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66798.bound LeftBound66804.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66798.bound, LeftBound66804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66798.actual selector witness) * (LeftBound66804.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66805

namespace LeftBound66813
def owner : Owner := ⟨.program ⟨214⟩, ⟨12761⟩⟩
def transferEvent : Nat := 66813
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66811 .coefficient, .predecessor 1 66812 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66811 .coefficient)
      LeftBound66803.bound (LeftBound66803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66803.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66812 .coefficient)
      LeftBound66775.bound (LeftBound66775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66775.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66803.bound, LeftBound66775.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66803.bound, LeftBound66775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66803.actual selector witness, LeftBound66775.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66813

namespace LeftBound66815
def owner : Owner := ⟨.program ⟨214⟩, ⟨12761⟩⟩
def transferEvent : Nat := 66815
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66810 .summary, .result 66780 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66810 .summary)
      LeftBound66805.bound (LeftBound66805.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10029⟩⟩) (rawTerms := some (Proof.Events260.exact66810RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66780 .summary)
      LeftBound66777.bound (LeftBound66777.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12760⟩⟩) (rawTerms := some (Proof.Events260.exact66780RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66777.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66805.bound, LeftBound66777.bound]
def bound : CoeffClass := .finite ⟨95458688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66805.bound, LeftBound66777.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66805.actual selector witness, LeftBound66777.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66815

namespace LeftBound66819
def owner : Owner := ⟨.program ⟨214⟩, ⟨25523⟩⟩
def transferEvent : Nat := 66819
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66817 .coefficient) (.predecessor 1 66818 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66817 .coefficient)
      LeftBound66813.bound (LeftBound66813.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66813.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66818 .coefficient)
      LeftAuthority66751.bound (LeftAuthority66751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events260.exact66752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66751.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66813.bound LeftAuthority66751.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66813.bound, LeftAuthority66751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66813.actual selector witness) * (LeftAuthority66751.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66819

namespace LeftBound66820
def owner : Owner := ⟨.program ⟨214⟩, ⟨25523⟩⟩
def transferEvent : Nat := 66820
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25522⟩⟩]⟩ [⟨.result 66752 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66752 .coefficient)
      LeftAuthority66751.bound (LeftAuthority66751.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25522⟩⟩) (rawTerms := some (Proof.Events260.exact66752RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66751.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66751.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66751.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66751.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66751.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66820

namespace LeftBound66821
def owner : Owner := ⟨.program ⟨214⟩, ⟨25523⟩⟩
def transferEvent : Nat := 66821
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66816 .summary) (.transfer 66820) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66816 .summary)
      LeftBound66815.bound (LeftBound66815.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12761⟩⟩) (rawTerms := some (Proof.Events261.exact66816RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66820)
      LeftBound66820.bound (LeftBound66820.actual selector witness) := by
  exact .transfer (LeftBound66820.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66815.bound LeftBound66820.bound
def bound : CoeffClass := .finite ⟨350334912299008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66815.bound, LeftBound66820.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66815.actual selector witness) * (LeftBound66820.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66821

namespace LeftBound66832
def owner : Owner := ⟨.program ⟨214⟩, ⟨20030⟩⟩
def transferEvent : Nat := 66832
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 66830 .coefficient) (.value (.predecessor 1 66831 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66830 .coefficient)
      LeftAuthority66828.bound (LeftAuthority66828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66828.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66831 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66828.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66828.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66828.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66832

namespace LeftBound66836
def owner : Owner := ⟨.program ⟨214⟩, ⟨20031⟩⟩
def transferEvent : Nat := 66836
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66834 .coefficient) (.predecessor 1 66835 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66834 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66835 .coefficient)
      LeftBound66832.bound (LeftBound66832.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66833RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66832.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66832.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound66832.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound66832.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound66832.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66836

namespace LeftBound66837
def owner : Owner := ⟨.program ⟨214⟩, ⟨20031⟩⟩
def transferEvent : Nat := 66837
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20028⟩⟩]⟩ [⟨.result 66829 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66829 .coefficient)
      LeftAuthority66828.bound (LeftAuthority66828.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20028⟩⟩) (rawTerms := some (Proof.Events261.exact66829RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66828.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66828.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66828.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66837

namespace LeftBound66838
def owner : Owner := ⟨.program ⟨214⟩, ⟨20031⟩⟩
def transferEvent : Nat := 66838
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 66837) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66837)
      LeftBound66837.bound (LeftBound66837.actual selector witness) := by
  exact .transfer (LeftBound66837.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound66837.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound66837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound66837.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66838

namespace LeftBound66917
def owner : Owner := ⟨.program ⟨214⟩, ⟨12755⟩⟩
def transferEvent : Nat := 66917
def frameStart : Nat := 66888
def rule : BoundRule := .product (.predecessor 0 66915 .coefficient) (.predecessor 1 66916 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66915 .coefficient)
      LeftAuthority66913.bound (LeftAuthority66913.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66914RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66913.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66913.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66916 .coefficient)
      LeftAuthority66910.bound (LeftAuthority66910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66910.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority66913.bound LeftAuthority66910.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66913.bound, LeftAuthority66910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority66913.actual selector witness) * (LeftAuthority66910.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66917

namespace LeftBound66921
def owner : Owner := ⟨.program ⟨214⟩, ⟨12756⟩⟩
def transferEvent : Nat := 66921
def frameStart : Nat := 66888
def rule : BoundRule := .identity (.predecessor 0 66920 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66920 .coefficient)
      LeftBound66917.bound (LeftBound66917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66917.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66917.derived selector witness)

def rawBound : CoeffClass := LeftBound66917.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66917.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66921

namespace LeftBound66938
def owner : Owner := ⟨.program ⟨214⟩, ⟨12854⟩⟩
def transferEvent : Nat := 66938
def frameStart : Nat := 66888
def rule : BoundRule := .sum [.predecessor 0 66936 .coefficient, .predecessor 1 66937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66936 .coefficient)
      LeftBound66921.bound (LeftBound66921.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66921.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66937 .coefficient)
      LeftAuthority66934.bound (LeftAuthority66934.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority66934.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66921.bound, LeftAuthority66934.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66921.bound, LeftAuthority66934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66921.actual selector witness, LeftAuthority66934.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66938

namespace LeftBound66941
def owner : Owner := ⟨.program ⟨214⟩, ⟨12855⟩⟩
def transferEvent : Nat := 66941
def frameStart : Nat := 66888
def rule : BoundRule := .identity (.predecessor 0 66940 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66940 .coefficient)
      LeftBound66938.bound (LeftBound66938.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound66938.derived selector witness)

def rawBound : CoeffClass := LeftBound66938.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66938.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound66938.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound66941

namespace LeftBound66947
def owner : Owner := ⟨.program ⟨214⟩, ⟨12856⟩⟩
def transferEvent : Nat := 66947
def frameStart : Nat := 66888
def rule : BoundRule := .product (.predecessor 0 66945 .coefficient) (.predecessor 1 66946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66945 .coefficient)
      LeftAuthority66943.bound (LeftAuthority66943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66943.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66946 .coefficient)
      LeftBound66941.bound (LeftBound66941.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact66942RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66941.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66941.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority66943.bound LeftBound66941.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66943.bound, LeftBound66941.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority66943.actual selector witness) * (LeftBound66941.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66947

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
