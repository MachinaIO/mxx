import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard047
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard109

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17898
def owner : Owner := ⟨.program ⟨214⟩, ⟨17736⟩⟩
def transferEvent : Nat := 17898
def frameStart : Nat := 17810
def rule : BoundRule := .product (.predecessor 0 17896 .coefficient) (.predecessor 1 17897 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17896 .coefficient)
      LeftAuthority17871.bound (LeftAuthority17871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17897 .coefficient)
      LeftAuthority17894.bound (LeftAuthority17894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority17871.bound LeftAuthority17894.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17871.bound, LeftAuthority17894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority17871.actual selector witness) * (LeftAuthority17894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17898

namespace LeftBound17906
def owner : Owner := ⟨.program ⟨214⟩, ⟨17737⟩⟩
def transferEvent : Nat := 17906
def frameStart : Nat := 17810
def rule : BoundRule := .sum [.predecessor 0 17904 .coefficient, .predecessor 1 17905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17904 .coefficient)
      LeftAuthority17902.bound (LeftAuthority17902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17905 .coefficient)
      LeftBound17898.bound (LeftBound17898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17902.bound, LeftBound17898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17902.bound, LeftBound17898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17902.actual selector witness, LeftBound17898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17906

namespace LeftBound17910
def owner : Owner := ⟨.program ⟨214⟩, ⟨29436⟩⟩
def transferEvent : Nat := 17910
def frameStart : Nat := 17810
def rule : BoundRule := .sum [.predecessor 0 17908 .coefficient, .predecessor 1 17909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17908 .coefficient)
      LeftBound17906.bound (LeftBound17906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17909 .coefficient)
      LeftBound17887.bound (LeftBound17887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17906.bound, LeftBound17887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17906.bound, LeftBound17887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17906.actual selector witness, LeftBound17887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17910

namespace LeftBound17923
def owner : Owner := ⟨.program ⟨214⟩, ⟨29433⟩⟩
def transferEvent : Nat := 17923
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 17921 .coefficient, .predecessor 1 17922 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17921 .coefficient)
      LeftBound17752.bound (LeftBound17752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17922 .coefficient)
      LeftBound17735.bound (LeftBound17735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17752.bound, LeftBound17735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17752.bound, LeftBound17735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17752.actual selector witness, LeftBound17735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17923

namespace LeftBound17926
def owner : Owner := ⟨.program ⟨214⟩, ⟨29433⟩⟩
def transferEvent : Nat := 17926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 17920 .summary, .result 17742 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17920 .summary)
      LeftBound17754.bound (LeftBound17754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22355⟩⟩) (rawTerms := some (Proof.Events070.exact17920RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17742 .summary)
      LeftBound17737.bound (LeftBound17737.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29432⟩⟩) (rawTerms := some (Proof.Events069.exact17742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17754.bound, LeftBound17737.bound]
def bound : CoeffClass := .finite ⟨1292382248169874534400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17754.bound, LeftBound17737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17754.actual selector witness, LeftBound17737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17926

namespace LeftBound17930
def owner : Owner := ⟨.program ⟨214⟩, ⟨29434⟩⟩
def transferEvent : Nat := 17930
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17928 .coefficient) (.predecessor 1 17929 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17928 .coefficient)
      LeftBound17923.bound (LeftBound17923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17929 .coefficient)
      LeftBound5578.bound (LeftBound5578.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5579RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5578.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5578.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17923.bound LeftBound5578.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17923.bound, LeftBound5578.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17923.actual selector witness) * (LeftBound5578.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17930

namespace LeftBound17931
def owner : Owner := ⟨.program ⟨214⟩, ⟨29434⟩⟩
def transferEvent : Nat := 17931
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6665⟩⟩]⟩ [⟨.result 5575 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5575 .coefficient)
      LeftAuthority5574.bound (LeftAuthority5574.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6665⟩⟩) (rawTerms := some (Proof.Events021.exact5575RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5574.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5574.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5574.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5574.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17931

namespace LeftBound17932
def owner : Owner := ⟨.program ⟨214⟩, ⟨29434⟩⟩
def transferEvent : Nat := 17932
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17927 .summary) (.transfer 17931) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17927 .summary)
      LeftBound17926.bound (LeftBound17926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29433⟩⟩) (rawTerms := some (Proof.Events070.exact17927RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17931)
      LeftBound17931.bound (LeftBound17931.actual selector witness) := by
  exact .transfer (LeftBound17931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17926.bound LeftBound17931.bound
def bound : CoeffClass := .finite ⟨4743063528899410259240550400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17926.bound, LeftBound17931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17926.actual selector witness) * (LeftBound17931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17932

namespace LeftBound17947
def owner : Owner := ⟨.program ⟨214⟩, ⟨29215⟩⟩
def transferEvent : Nat := 17947
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17945 .coefficient) (.predecessor 1 17946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17945 .coefficient)
      LeftBound8747.bound (LeftBound8747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events034.exact8751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17946 .coefficient)
      LeftAuthority17943.bound (LeftAuthority17943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8747.bound LeftAuthority17943.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8747.bound, LeftAuthority17943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8747.actual selector witness) * (LeftAuthority17943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17947

namespace LeftBound17948
def owner : Owner := ⟨.program ⟨214⟩, ⟨29215⟩⟩
def transferEvent : Nat := 17948
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29213⟩⟩]⟩ [⟨.result 17944 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17944 .coefficient)
      LeftAuthority17943.bound (LeftAuthority17943.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29213⟩⟩) (rawTerms := some (Proof.Events070.exact17944RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17943.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17943.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17943.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17948

namespace LeftBound17949
def owner : Owner := ⟨.program ⟨214⟩, ⟨29215⟩⟩
def transferEvent : Nat := 17949
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8751 .summary) (.transfer 17948) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8751 .summary)
      LeftBound8750.bound (LeftBound8750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25472⟩⟩) (rawTerms := some (Proof.Events034.exact8751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17948)
      LeftBound17948.bound (LeftBound17948.actual selector witness) := by
  exact .transfer (LeftBound17948.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8750.bound LeftBound17948.bound
def bound : CoeffClass := .finite ⟨1292337421468529852416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8750.bound, LeftBound17948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8750.actual selector witness) * (LeftBound17948.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17949

namespace LeftBound17960
def owner : Owner := ⟨.program ⟨214⟩, ⟨22210⟩⟩
def transferEvent : Nat := 17960
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 17958 .coefficient) (.value (.predecessor 1 17959 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17958 .coefficient)
      LeftAuthority17956.bound (LeftAuthority17956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17959 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority17956.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17956.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17956.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound17960

namespace LeftBound17964
def owner : Owner := ⟨.program ⟨214⟩, ⟨22211⟩⟩
def transferEvent : Nat := 17964
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17962 .coefficient) (.predecessor 1 17963 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17962 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17963 .coefficient)
      LeftBound17960.bound (LeftBound17960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact17961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound17960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound17960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound17960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17964

namespace LeftBound17965
def owner : Owner := ⟨.program ⟨214⟩, ⟨22211⟩⟩
def transferEvent : Nat := 17965
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22208⟩⟩]⟩ [⟨.result 17957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17957 .coefficient)
      LeftAuthority17956.bound (LeftAuthority17956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22208⟩⟩) (rawTerms := some (Proof.Events070.exact17957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17956.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17965

namespace LeftBound17966
def owner : Owner := ⟨.program ⟨214⟩, ⟨22211⟩⟩
def transferEvent : Nat := 17966
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 17965) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17965)
      LeftBound17965.bound (LeftBound17965.actual selector witness) := by
  exact .transfer (LeftBound17965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound17965.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound17965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound17965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17966

namespace LeftBound18061
def owner : Owner := ⟨.program ⟨214⟩, ⟨16566⟩⟩
def transferEvent : Nat := 18061
def frameStart : Nat := 18022
def rule : BoundRule := .identity (.predecessor 0 18060 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18060 .coefficient)
      LeftAuthority18058.bound (LeftAuthority18058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events070.exact18059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18058.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18058.derived selector witness)

def rawBound : CoeffClass := LeftAuthority18058.bound
def bound : CoeffClass := .finite ⟨42, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority18058.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18061

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
