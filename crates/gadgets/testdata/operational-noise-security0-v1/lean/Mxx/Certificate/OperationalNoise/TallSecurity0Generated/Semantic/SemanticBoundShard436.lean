import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard022
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard023
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard336
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard409
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard435

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64904
def owner : Owner := ⟨.program ⟨214⟩, ⟨30148⟩⟩
def transferEvent : Nat := 64904
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64900 .summary, .result 60967 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64900 .summary)
      LeftBound64899.bound (LeftBound64899.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30137⟩⟩) (rawTerms := some (Proof.Events253.exact64900RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64899.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 60967 .summary)
      LeftBound60962.bound (LeftBound60962.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30146⟩⟩) (rawTerms := some (Proof.Events238.exact60967RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound60962.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64899.bound, LeftBound60962.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789388161076, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64899.bound, LeftBound60962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64899.actual selector witness, LeftBound60962.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64904

namespace LeftBound64910
def owner : Owner := ⟨.program ⟨214⟩, ⟨7092⟩⟩
def transferEvent : Nat := 64910
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64908 .coefficient) (.predecessor 1 64909 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64908 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64909 .coefficient)
      LeftAuthority6083.bound (LeftAuthority6083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6083.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6083.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound26.bound LeftAuthority6083.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftAuthority6083.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound26.actual selector witness) * (LeftAuthority6083.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64910

namespace LeftBound64915
def owner : Owner := ⟨.program ⟨214⟩, ⟨7725⟩⟩
def transferEvent : Nat := 64915
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64913 .coefficient, .predecessor 1 64914 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64913 .coefficient)
      LeftBound64910.bound (LeftBound64910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64914 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64910.bound, LeftBound50668.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64910.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64910.actual selector witness, LeftBound50668.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64915

namespace LeftBound64919
def owner : Owner := ⟨.program ⟨214⟩, ⟨7726⟩⟩
def transferEvent : Nat := 64919
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64917 .coefficient, .predecessor 1 64918 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64917 .coefficient)
      LeftBound64915.bound (LeftBound64915.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64915.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64915.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64918 .coefficient)
      LeftAuthority64906.bound (LeftAuthority64906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64915.bound, LeftAuthority64906.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64915.bound, LeftAuthority64906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64915.actual selector witness, LeftAuthority64906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64919

namespace LeftBound64920
def owner : Owner := ⟨.program ⟨214⟩, ⟨7726⟩⟩
def transferEvent : Nat := 64920
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨68⟩⟩]⟩ [⟨.result 64907 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64907 .coefficient)
      LeftAuthority64906.bound (LeftAuthority64906.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨68⟩⟩) (rawTerms := some (Proof.Events253.exact64907RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64906.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64906.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64906.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64906.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64920

namespace LeftBound64925
def owner : Owner := ⟨.program ⟨214⟩, ⟨7902⟩⟩
def transferEvent : Nat := 64925
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64923 .coefficient) (.predecessor 1 64924 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64923 .coefficient)
      LeftBound64919.bound (LeftBound64919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64922RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64924 .coefficient)
      LeftBound5960.bound (LeftBound5960.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact5961RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5960.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5960.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64919.bound LeftBound5960.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64919.bound, LeftBound5960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64919.actual selector witness) * (LeftBound5960.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64925

namespace LeftBound64926
def owner : Owner := ⟨.program ⟨214⟩, ⟨7902⟩⟩
def transferEvent : Nat := 64926
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7885⟩⟩]⟩ [⟨.result 5957 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5957 .coefficient)
      LeftAuthority5956.bound (LeftAuthority5956.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7885⟩⟩) (rawTerms := some (Proof.Events023.exact5957RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5956.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5956.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5956.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5956.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5956.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64926

namespace LeftBound64927
def owner : Owner := ⟨.program ⟨214⟩, ⟨7902⟩⟩
def transferEvent : Nat := 64927
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64922 .summary) (.transfer 64926) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64922 .summary)
      LeftBound64920.bound (LeftBound64920.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7726⟩⟩) (rawTerms := some (Proof.Events253.exact64922RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64926)
      LeftBound64926.bound (LeftBound64926.actual selector witness) := by
  exact .transfer (LeftBound64926.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64920.bound LeftBound64926.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64920.bound, LeftBound64926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64920.actual selector witness) * (LeftBound64926.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64927

namespace LeftBound64953
def owner : Owner := ⟨.program ⟨214⟩, ⟨30149⟩⟩
def transferEvent : Nat := 64953
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64951 .coefficient, .predecessor 1 64952 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64951 .coefficient)
      LeftBound64925.bound (LeftBound64925.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64950RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64925.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64925.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64952 .coefficient)
      LeftBound64903.bound (LeftBound64903.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64905RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64903.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64903.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64925.bound, LeftBound64903.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64925.bound, LeftBound64903.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64925.actual selector witness, LeftBound64903.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64953

namespace LeftBound64973
def owner : Owner := ⟨.program ⟨214⟩, ⟨30149⟩⟩
def transferEvent : Nat := 64973
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 64950 .summary, .result 64905 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64950 .summary)
      LeftBound64927.bound (LeftBound64927.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨7902⟩⟩) (rawTerms := some (Proof.Events253.exact64950RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64927.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64905 .summary)
      LeftBound64904.bound (LeftBound64904.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30148⟩⟩) (rawTerms := some (Proof.Events253.exact64905RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64904.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64927.bound, LeftBound64904.bound]
def bound : CoeffClass := .finite ⟨313276456757822654825721789483581492, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64927.bound, LeftBound64904.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64927.actual selector witness, LeftBound64904.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64973

namespace LeftBound64977
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def transferEvent : Nat := 64977
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64975 .coefficient) (.predecessor 1 64976 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64975 .coefficient)
      LeftBound64953.bound (LeftBound64953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events253.exact64974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64953.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64976 .coefficient)
      LeftBound6080.bound (LeftBound6080.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6081RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6080.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6080.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64953.bound LeftBound6080.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64953.bound, LeftBound6080.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64953.actual selector witness) * (LeftBound6080.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64977

namespace LeftBound64978
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def transferEvent : Nat := 64978
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7825⟩⟩]⟩ [⟨.result 6077 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6077 .coefficient)
      LeftAuthority6076.bound (LeftAuthority6076.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7825⟩⟩) (rawTerms := some (Proof.Events023.exact6077RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6076.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6076.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6076.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6076.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6076.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64978

namespace LeftBound64979
def owner : Owner := ⟨.program ⟨214⟩, ⟨30150⟩⟩
def transferEvent : Nat := 64979
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 64974 .summary) (.transfer 64978) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64974 .summary)
      LeftBound64973.bound (LeftBound64973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30149⟩⟩) (rawTerms := some (Proof.Events253.exact64974RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64973.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64978)
      LeftBound64978.bound (LeftBound64978.actual selector witness) := by
  exact .transfer (LeftBound64978.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64973.bound LeftBound64978.bound
def bound : CoeffClass := .finite ⟨1149729608724517268372876178953375812943872, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64973.bound, LeftBound64978.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64973.actual selector witness) * (LeftBound64978.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64979

namespace LeftBound65041
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def transferEvent : Nat := 65041
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65039 .coefficient, .predecessor 1 65040 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65039 .coefficient)
      LeftBound64977.bound (LeftBound64977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65038RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65040 .coefficient)
      LeftBound50558.bound (LeftBound50558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64977.bound, LeftBound50558.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64977.bound, LeftBound50558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64977.actual selector witness, LeftBound50558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65041

namespace LeftBound65061
def owner : Owner := ⟨.program ⟨214⟩, ⟨30151⟩⟩
def transferEvent : Nat := 65061
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 65038 .summary, .result 50635 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65038 .summary)
      LeftBound64979.bound (LeftBound64979.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30150⟩⟩) (rawTerms := some (Proof.Events254.exact65038RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound64979.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50635 .summary)
      LeftBound50596.bound (LeftBound50596.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨18861⟩⟩) (rawTerms := some (Proof.Events197.exact50635RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50596.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64979.bound, LeftBound50596.bound]
def bound : CoeffClass := .finite ⟨1149729608724524008718218297164355856419136, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64979.bound, LeftBound50596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64979.actual selector witness, LeftBound50596.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65061

namespace LeftBound65065
def owner : Owner := ⟨.program ⟨214⟩, ⟨30152⟩⟩
def transferEvent : Nat := 65065
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65063 .coefficient) (.predecessor 1 65064 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 65063 .coefficient)
      LeftBound65041.bound (LeftBound65041.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65062RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65041.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65041.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 65064 .coefficient)
      LeftBound6070.bound (LeftBound6070.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events023.exact6071RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6070.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6070.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound65041.bound LeftBound6070.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65041.bound, LeftBound6070.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound65041.actual selector witness) * (LeftBound6070.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65065

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
