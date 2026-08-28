import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard373
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard420

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound62844
def owner : Owner := ⟨.program ⟨214⟩, ⟨17669⟩⟩
def transferEvent : Nat := 62844
def frameStart : Nat := 62748
def rule : BoundRule := .sum [.predecessor 0 62842 .coefficient, .predecessor 1 62843 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62842 .coefficient)
      LeftAuthority62840.bound (LeftAuthority62840.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62840.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62840.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62843 .coefficient)
      LeftBound62836.bound (LeftBound62836.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62838RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62836.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62836.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority62840.bound, LeftBound62836.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62840.bound, LeftBound62836.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority62840.actual selector witness, LeftBound62836.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62844

namespace LeftBound62848
def owner : Owner := ⟨.program ⟨214⟩, ⟨28312⟩⟩
def transferEvent : Nat := 62848
def frameStart : Nat := 62748
def rule : BoundRule := .sum [.predecessor 0 62846 .coefficient, .predecessor 1 62847 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62846 .coefficient)
      LeftBound62844.bound (LeftBound62844.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62845RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62844.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62844.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62847 .coefficient)
      LeftBound62825.bound (LeftBound62825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62830RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62825.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62844.bound, LeftBound62825.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62844.bound, LeftBound62825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62844.actual selector witness, LeftBound62825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62848

namespace LeftBound62861
def owner : Owner := ⟨.program ⟨214⟩, ⟨28309⟩⟩
def transferEvent : Nat := 62861
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 62859 .coefficient, .predecessor 1 62860 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62859 .coefficient)
      LeftBound62690.bound (LeftBound62690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62858RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62860 .coefficient)
      LeftBound62673.bound (LeftBound62673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events244.exact62680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62690.bound, LeftBound62673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62690.bound, LeftBound62673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62690.actual selector witness, LeftBound62673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62861

namespace LeftBound62864
def owner : Owner := ⟨.program ⟨214⟩, ⟨28309⟩⟩
def transferEvent : Nat := 62864
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 62858 .summary, .result 62680 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62858 .summary)
      LeftBound62692.bound (LeftBound62692.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21623⟩⟩) (rawTerms := some (Proof.Events245.exact62858RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62680 .summary)
      LeftBound62675.bound (LeftBound62675.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28308⟩⟩) (rawTerms := some (Proof.Events244.exact62680RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62692.bound, LeftBound62675.bound]
def bound : CoeffClass := .finite ⟨1292180536164689260544, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62692.bound, LeftBound62675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62692.actual selector witness, LeftBound62675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound62864

namespace LeftBound62868
def owner : Owner := ⟨.program ⟨214⟩, ⟨28310⟩⟩
def transferEvent : Nat := 62868
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62866 .coefficient) (.predecessor 1 62867 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62866 .coefficient)
      LeftBound62861.bound (LeftBound62861.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62865RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62861.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62861.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62867 .coefficient)
      LeftBound5678.bound (LeftBound5678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62861.bound LeftBound5678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62861.bound, LeftBound5678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62861.actual selector witness) * (LeftBound5678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62868

namespace LeftBound62869
def owner : Owner := ⟨.program ⟨214⟩, ⟨28310⟩⟩
def transferEvent : Nat := 62869
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6681⟩⟩]⟩ [⟨.result 5675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5675 .coefficient)
      LeftAuthority5674.bound (LeftAuthority5674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6681⟩⟩) (rawTerms := some (Proof.Events022.exact5675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5674.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62869

namespace LeftBound62870
def owner : Owner := ⟨.program ⟨214⟩, ⟨28310⟩⟩
def transferEvent : Nat := 62870
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 62865 .summary) (.transfer 62869) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62865 .summary)
      LeftBound62864.bound (LeftBound62864.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28309⟩⟩) (rawTerms := some (Proof.Events245.exact62865RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound62864.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62869)
      LeftBound62869.bound (LeftBound62869.actual selector witness) := by
  exact .transfer (LeftBound62869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound62864.bound LeftBound62869.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62864.bound, LeftBound62869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound62864.actual selector witness) * (LeftBound62869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62870

namespace LeftBound62885
def owner : Owner := ⟨.program ⟨214⟩, ⟨28091⟩⟩
def transferEvent : Nat := 62885
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62883 .coefficient) (.predecessor 1 62884 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62883 .coefficient)
      LeftBound55282.bound (LeftBound55282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events215.exact55286RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound55282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound55282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62884 .coefficient)
      LeftAuthority62881.bound (LeftAuthority62881.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62882RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62881.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55282.bound LeftAuthority62881.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55282.bound, LeftAuthority62881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55282.actual selector witness) * (LeftAuthority62881.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62885

namespace LeftBound62886
def owner : Owner := ⟨.program ⟨214⟩, ⟨28091⟩⟩
def transferEvent : Nat := 62886
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28089⟩⟩]⟩ [⟨.result 62882 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62882 .coefficient)
      LeftAuthority62881.bound (LeftAuthority62881.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28089⟩⟩) (rawTerms := some (Proof.Events245.exact62882RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62881.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62881.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62881.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62881.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62881.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62886

namespace LeftBound62887
def owner : Owner := ⟨.program ⟨214⟩, ⟨28091⟩⟩
def transferEvent : Nat := 62887
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 55286 .summary) (.transfer 62886) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 55286 .summary)
      LeftBound55285.bound (LeftBound55285.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26150⟩⟩) (rawTerms := some (Proof.Events215.exact55286RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound55285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62886)
      LeftBound62886.bound (LeftBound62886.actual selector witness) := by
  exact .transfer (LeftBound62886.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound55285.bound LeftBound62886.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound55285.bound, LeftBound62886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound55285.actual selector witness) * (LeftBound62886.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62887

namespace LeftBound62898
def owner : Owner := ⟨.program ⟨214⟩, ⟨21478⟩⟩
def transferEvent : Nat := 62898
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 62896 .coefficient) (.value (.predecessor 1 62897 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62896 .coefficient)
      LeftAuthority62894.bound (LeftAuthority62894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62894.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62897 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority62894.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62894.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62894.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound62898

namespace LeftBound62902
def owner : Owner := ⟨.program ⟨214⟩, ⟨21479⟩⟩
def transferEvent : Nat := 62902
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 62900 .coefficient) (.predecessor 1 62901 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62900 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 62901 .coefficient)
      LeftBound62898.bound (LeftBound62898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events245.exact62899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound62898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound62898.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound62898.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound62898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound62898.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62902

namespace LeftBound62903
def owner : Owner := ⟨.program ⟨214⟩, ⟨21479⟩⟩
def transferEvent : Nat := 62903
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21476⟩⟩]⟩ [⟨.result 62895 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 62895 .coefficient)
      LeftAuthority62894.bound (LeftAuthority62894.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21476⟩⟩) (rawTerms := some (Proof.Events245.exact62895RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62894.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority62894.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority62894.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound62903

namespace LeftBound62904
def owner : Owner := ⟨.program ⟨214⟩, ⟨21479⟩⟩
def transferEvent : Nat := 62904
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 62903) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 62903)
      LeftBound62903.bound (LeftBound62903.actual selector witness) := by
  exact .transfer (LeftBound62903.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound62903.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound62903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound62903.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound62904

namespace LeftBound62999
def owner : Owner := ⟨.program ⟨214⟩, ⟨16064⟩⟩
def transferEvent : Nat := 62999
def frameStart : Nat := 62960
def rule : BoundRule := .identity (.predecessor 0 62998 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 62998 .coefficient)
      LeftAuthority62996.bound (LeftAuthority62996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact62997RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority62996.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority62996.derived selector witness)

def rawBound : CoeffClass := LeftAuthority62996.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority62996.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority62996.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound62999

namespace LeftBound63016
def owner : Owner := ⟨.program ⟨214⟩, ⟨16138⟩⟩
def transferEvent : Nat := 63016
def frameStart : Nat := 62960
def rule : BoundRule := .sum [.predecessor 0 63014 .coefficient, .predecessor 1 63015 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63014 .coefficient)
      LeftBound62999.bound (LeftBound62999.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound62999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63015 .coefficient)
      LeftAuthority63012.bound (LeftAuthority63012.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63012.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound62999.bound, LeftAuthority63012.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound62999.bound, LeftAuthority63012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound62999.actual selector witness, LeftAuthority63012.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63016

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
