import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard591
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard628

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92887
def owner : Owner := ⟨.program ⟨214⟩, ⟨27426⟩⟩
def transferEvent : Nat := 92887
def frameStart : Nat := 92810
def rule : BoundRule := .product (.predecessor 0 92885 .coefficient) (.predecessor 1 92886 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92885 .coefficient)
      LeftBound92883.bound (LeftBound92883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92886 .coefficient)
      LeftAuthority92860.bound (LeftAuthority92860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92883.bound LeftAuthority92860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92883.bound, LeftAuthority92860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92883.actual selector witness) * (LeftAuthority92860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92887

namespace LeftBound92898
def owner : Owner := ⟨.program ⟨214⟩, ⟨17440⟩⟩
def transferEvent : Nat := 92898
def frameStart : Nat := 92810
def rule : BoundRule := .product (.predecessor 0 92896 .coefficient) (.predecessor 1 92897 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92896 .coefficient)
      LeftAuthority92871.bound (LeftAuthority92871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92897 .coefficient)
      LeftAuthority92894.bound (LeftAuthority92894.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92895RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92894.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92871.bound LeftAuthority92894.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92871.bound, LeftAuthority92894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority92871.actual selector witness) * (LeftAuthority92894.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92898

namespace LeftBound92906
def owner : Owner := ⟨.program ⟨214⟩, ⟨17441⟩⟩
def transferEvent : Nat := 92906
def frameStart : Nat := 92810
def rule : BoundRule := .sum [.predecessor 0 92904 .coefficient, .predecessor 1 92905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92904 .coefficient)
      LeftAuthority92902.bound (LeftAuthority92902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92905 .coefficient)
      LeftBound92898.bound (LeftBound92898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92902.bound, LeftBound92898.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92902.bound, LeftBound92898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92902.actual selector witness, LeftBound92898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92906

namespace LeftBound92910
def owner : Owner := ⟨.program ⟨214⟩, ⟨27431⟩⟩
def transferEvent : Nat := 92910
def frameStart : Nat := 92810
def rule : BoundRule := .sum [.predecessor 0 92908 .coefficient, .predecessor 1 92909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92908 .coefficient)
      LeftBound92906.bound (LeftBound92906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92909 .coefficient)
      LeftBound92887.bound (LeftBound92887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92887.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92906.bound, LeftBound92887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92906.bound, LeftBound92887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92906.actual selector witness, LeftBound92887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92910

namespace LeftBound92923
def owner : Owner := ⟨.program ⟨214⟩, ⟨27428⟩⟩
def transferEvent : Nat := 92923
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92921 .coefficient, .predecessor 1 92922 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92921 .coefficient)
      LeftBound92752.bound (LeftBound92752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92922 .coefficient)
      LeftBound92735.bound (LeftBound92735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92742RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92735.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92752.bound, LeftBound92735.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92752.bound, LeftBound92735.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92752.actual selector witness, LeftBound92735.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92923

namespace LeftBound92926
def owner : Owner := ⟨.program ⟨214⟩, ⟨27428⟩⟩
def transferEvent : Nat := 92926
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 92920 .summary, .result 92742 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92920 .summary)
      LeftBound92754.bound (LeftBound92754.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21043⟩⟩) (rawTerms := some (Proof.Events362.exact92920RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92754.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92742 .summary)
      LeftBound92737.bound (LeftBound92737.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27427⟩⟩) (rawTerms := some (Proof.Events362.exact92742RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92737.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92754.bound, LeftBound92737.bound]
def bound : CoeffClass := .finite ⟨1292001236604524572672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92754.bound, LeftBound92737.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92754.actual selector witness, LeftBound92737.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92926

namespace LeftBound92930
def owner : Owner := ⟨.program ⟨214⟩, ⟨27429⟩⟩
def transferEvent : Nat := 92930
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92928 .coefficient) (.predecessor 1 92929 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92928 .coefficient)
      LeftBound92923.bound (LeftBound92923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92923.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92923.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92929 .coefficient)
      LeftBound5758.bound (LeftBound5758.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5759RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5758.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5758.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92923.bound LeftBound5758.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92923.bound, LeftBound5758.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92923.actual selector witness) * (LeftBound5758.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92930

namespace LeftBound92931
def owner : Owner := ⟨.program ⟨214⟩, ⟨27429⟩⟩
def transferEvent : Nat := 92931
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6647⟩⟩]⟩ [⟨.result 5755 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5755 .coefficient)
      LeftAuthority5754.bound (LeftAuthority5754.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6647⟩⟩) (rawTerms := some (Proof.Events022.exact5755RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5754.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5754.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5754.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5754.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92931

namespace LeftBound92932
def owner : Owner := ⟨.program ⟨214⟩, ⟨27429⟩⟩
def transferEvent : Nat := 92932
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 92927 .summary) (.transfer 92931) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92927 .summary)
      LeftBound92926.bound (LeftBound92926.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27428⟩⟩) (rawTerms := some (Proof.Events362.exact92927RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound92926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92931)
      LeftBound92931.bound (LeftBound92931.actual selector witness) := by
  exact .transfer (LeftBound92931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92926.bound LeftBound92931.bound
def bound : CoeffClass := .finite ⟨4741665210358390854099402752, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92926.bound, LeftBound92931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92926.actual selector witness) * (LeftBound92931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92932

namespace LeftBound92947
def owner : Owner := ⟨.program ⟨214⟩, ⟨27210⟩⟩
def transferEvent : Nat := 92947
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92945 .coefficient) (.predecessor 1 92946 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92945 .coefficient)
      LeftBound86432.bound (LeftBound86432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86436RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92946 .coefficient)
      LeftAuthority92943.bound (LeftAuthority92943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92943.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86432.bound LeftAuthority92943.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86432.bound, LeftAuthority92943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86432.actual selector witness) * (LeftAuthority92943.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92947

namespace LeftBound92948
def owner : Owner := ⟨.program ⟨214⟩, ⟨27210⟩⟩
def transferEvent : Nat := 92948
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27208⟩⟩]⟩ [⟨.result 92944 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92944 .coefficient)
      LeftAuthority92943.bound (LeftAuthority92943.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27208⟩⟩) (rawTerms := some (Proof.Events363.exact92944RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92943.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92943.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92943.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92948

namespace LeftBound92949
def owner : Owner := ⟨.program ⟨214⟩, ⟨27210⟩⟩
def transferEvent : Nat := 92949
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86436 .summary) (.transfer 92948) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86436 .summary)
      LeftBound86435.bound (LeftBound86435.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25837⟩⟩) (rawTerms := some (Proof.Events337.exact86436RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86435.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92948)
      LeftBound92948.bound (LeftBound92948.actual selector witness) := by
  exact .transfer (LeftBound92948.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86435.bound LeftBound92948.bound
def bound : CoeffClass := .finite ⟨1291978822348200476672, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86435.bound, LeftBound92948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86435.actual selector witness) * (LeftBound92948.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92949

namespace LeftBound92960
def owner : Owner := ⟨.program ⟨214⟩, ⟨20898⟩⟩
def transferEvent : Nat := 92960
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92958 .coefficient) (.value (.predecessor 1 92959 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92958 .coefficient)
      LeftAuthority92956.bound (LeftAuthority92956.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92957RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92956.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92959 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92956.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92956.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92956.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92960

namespace LeftBound92964
def owner : Owner := ⟨.program ⟨214⟩, ⟨20899⟩⟩
def transferEvent : Nat := 92964
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92962 .coefficient) (.predecessor 1 92963 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92962 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92963 .coefficient)
      LeftBound92960.bound (LeftBound92960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events363.exact92961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound92960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound92960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound92960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92964

namespace LeftBound92965
def owner : Owner := ⟨.program ⟨214⟩, ⟨20899⟩⟩
def transferEvent : Nat := 92965
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20896⟩⟩]⟩ [⟨.result 92957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92957 .coefficient)
      LeftAuthority92956.bound (LeftAuthority92956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20896⟩⟩) (rawTerms := some (Proof.Events363.exact92957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92956.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92965

namespace LeftBound92966
def owner : Owner := ⟨.program ⟨214⟩, ⟨20899⟩⟩
def transferEvent : Nat := 92966
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 92965) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92965)
      LeftBound92965.bound (LeftBound92965.actual selector witness) := by
  exact .transfer (LeftBound92965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound92965.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound92965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound92965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92966

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
