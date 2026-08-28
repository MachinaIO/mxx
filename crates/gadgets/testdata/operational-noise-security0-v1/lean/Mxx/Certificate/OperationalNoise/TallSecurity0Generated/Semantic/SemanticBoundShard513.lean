import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard449
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard512

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound75954
def owner : Owner := ⟨.program ⟨214⟩, ⟨16965⟩⟩
def transferEvent : Nat := 75954
def frameStart : Nat := 75889
def rule : BoundRule := .product (.predecessor 0 75952 .coefficient) (.predecessor 1 75953 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75952 .coefficient)
      LeftAuthority75950.bound (LeftAuthority75950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75953 .coefficient)
      LeftBound75948.bound (LeftBound75948.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75949RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75948.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75948.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority75950.bound LeftBound75948.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75950.bound, LeftBound75948.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority75950.actual selector witness) * (LeftBound75948.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75954

namespace LeftBound75962
def owner : Owner := ⟨.program ⟨214⟩, ⟨16966⟩⟩
def transferEvent : Nat := 75962
def frameStart : Nat := 75889
def rule : BoundRule := .sum [.predecessor 0 75960 .coefficient, .predecessor 1 75961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75960 .coefficient)
      LeftAuthority75958.bound (LeftAuthority75958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75958.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75961 .coefficient)
      LeftBound75954.bound (LeftBound75954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75956RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75954.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75958.bound, LeftBound75954.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75958.bound, LeftBound75954.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75958.actual selector witness, LeftBound75954.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75962

namespace LeftBound75966
def owner : Owner := ⟨.program ⟨214⟩, ⟨29800⟩⟩
def transferEvent : Nat := 75966
def frameStart : Nat := 75889
def rule : BoundRule := .product (.predecessor 0 75964 .coefficient) (.predecessor 1 75965 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75964 .coefficient)
      LeftBound75962.bound (LeftBound75962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75965 .coefficient)
      LeftAuthority75939.bound (LeftAuthority75939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75939.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound75962.bound LeftAuthority75939.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75962.bound, LeftAuthority75939.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound75962.actual selector witness) * (LeftAuthority75939.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75966

namespace LeftBound75977
def owner : Owner := ⟨.program ⟨214⟩, ⟨16925⟩⟩
def transferEvent : Nat := 75977
def frameStart : Nat := 75889
def rule : BoundRule := .product (.predecessor 0 75975 .coefficient) (.predecessor 1 75976 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75975 .coefficient)
      LeftAuthority75950.bound (LeftAuthority75950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75976 .coefficient)
      LeftAuthority75973.bound (LeftAuthority75973.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75974RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75973.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75973.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority75950.bound LeftAuthority75973.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75950.bound, LeftAuthority75973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority75950.actual selector witness) * (LeftAuthority75973.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound75977

namespace LeftBound75985
def owner : Owner := ⟨.program ⟨214⟩, ⟨16926⟩⟩
def transferEvent : Nat := 75985
def frameStart : Nat := 75889
def rule : BoundRule := .sum [.predecessor 0 75983 .coefficient, .predecessor 1 75984 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75983 .coefficient)
      LeftAuthority75981.bound (LeftAuthority75981.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75982RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority75981.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority75981.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75984 .coefficient)
      LeftBound75977.bound (LeftBound75977.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75977.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75977.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority75981.bound, LeftBound75977.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority75981.bound, LeftBound75977.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority75981.actual selector witness, LeftBound75977.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75985

namespace LeftBound75989
def owner : Owner := ⟨.program ⟨214⟩, ⟨29805⟩⟩
def transferEvent : Nat := 75989
def frameStart : Nat := 75889
def rule : BoundRule := .sum [.predecessor 0 75987 .coefficient, .predecessor 1 75988 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 75987 .coefficient)
      LeftBound75985.bound (LeftBound75985.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75985.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75985.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 75988 .coefficient)
      LeftBound75966.bound (LeftBound75966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75985.bound, LeftBound75966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75985.bound, LeftBound75966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75985.actual selector witness, LeftBound75966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound75989

namespace LeftBound76002
def owner : Owner := ⟨.program ⟨214⟩, ⟨29802⟩⟩
def transferEvent : Nat := 76002
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 76000 .coefficient, .predecessor 1 76001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76000 .coefficient)
      LeftBound75831.bound (LeftBound75831.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75831.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75831.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76001 .coefficient)
      LeftBound75814.bound (LeftBound75814.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact75821RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound75814.bound, RecordedBoundRefines] <;> decide)
      (LeftBound75814.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75831.bound, LeftBound75814.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75831.bound, LeftBound75814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75831.actual selector witness, LeftBound75814.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76002

namespace LeftBound76005
def owner : Owner := ⟨.program ⟨214⟩, ⟨29802⟩⟩
def transferEvent : Nat := 76005
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 75999 .summary, .result 75821 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75999 .summary)
      LeftBound75833.bound (LeftBound75833.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22623⟩⟩) (rawTerms := some (Proof.Events296.exact75999RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75833.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 75821 .summary)
      LeftBound75816.bound (LeftBound75816.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29801⟩⟩) (rawTerms := some (Proof.Events296.exact75821RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound75816.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound75833.bound, LeftBound75816.bound]
def bound : CoeffClass := .finite ⟨1292516722839998050304, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound75833.bound, LeftBound75816.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound75833.actual selector witness, LeftBound75816.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76005

namespace LeftBound76009
def owner : Owner := ⟨.program ⟨214⟩, ⟨29803⟩⟩
def transferEvent : Nat := 76009
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76007 .coefficient) (.predecessor 1 76008 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76007 .coefficient)
      LeftBound76002.bound (LeftBound76002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact76006RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76008 .coefficient)
      LeftBound5538.bound (LeftBound5538.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5539RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5538.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5538.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76002.bound LeftBound5538.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76002.bound, LeftBound5538.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76002.actual selector witness) * (LeftBound5538.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76009

namespace LeftBound76010
def owner : Owner := ⟨.program ⟨214⟩, ⟨29803⟩⟩
def transferEvent : Nat := 76010
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6659⟩⟩]⟩ [⟨.result 5535 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5535 .coefficient)
      LeftAuthority5534.bound (LeftAuthority5534.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6659⟩⟩) (rawTerms := some (Proof.Events021.exact5535RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5534.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5534.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5534.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5534.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76010

namespace LeftBound76011
def owner : Owner := ⟨.program ⟨214⟩, ⟨29803⟩⟩
def transferEvent : Nat := 76011
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 76006 .summary) (.transfer 76010) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76006 .summary)
      LeftBound76005.bound (LeftBound76005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29802⟩⟩) (rawTerms := some (Proof.Events296.exact76006RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound76005.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76010)
      LeftBound76010.bound (LeftBound76010.actual selector witness) := by
  exact .transfer (LeftBound76010.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76005.bound LeftBound76010.bound
def bound : CoeffClass := .finite ⟨4743557053090358284584484864, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76005.bound, LeftBound76010.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76005.actual selector witness) * (LeftBound76010.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76011

namespace LeftBound76026
def owner : Owner := ⟨.program ⟨214⟩, ⟨29584⟩⟩
def transferEvent : Nat := 76026
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76024 .coefficient) (.predecessor 1 76025 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76024 .coefficient)
      LeftBound66533.bound (LeftBound66533.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66533.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66533.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76025 .coefficient)
      LeftAuthority76022.bound (LeftAuthority76022.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events296.exact76023RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66533.bound LeftAuthority76022.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66533.bound, LeftAuthority76022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66533.actual selector witness) * (LeftAuthority76022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76026

namespace LeftBound76027
def owner : Owner := ⟨.program ⟨214⟩, ⟨29584⟩⟩
def transferEvent : Nat := 76027
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29582⟩⟩]⟩ [⟨.result 76023 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76023 .coefficient)
      LeftAuthority76022.bound (LeftAuthority76022.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29582⟩⟩) (rawTerms := some (Proof.Events296.exact76023RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76022.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76022.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76022.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76022.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76027

namespace LeftBound76028
def owner : Owner := ⟨.program ⟨214⟩, ⟨29584⟩⟩
def transferEvent : Nat := 76028
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66537 .summary) (.transfer 76027) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66537 .summary)
      LeftBound66536.bound (LeftBound66536.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25601⟩⟩) (rawTerms := some (Proof.Events259.exact66537RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76027)
      LeftBound76027.bound (LeftBound76027.actual selector witness) := by
  exact .transfer (LeftBound76027.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66536.bound LeftBound76027.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66536.bound, LeftBound76027.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66536.actual selector witness) * (LeftBound76027.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76028

namespace LeftBound76039
def owner : Owner := ⟨.program ⟨214⟩, ⟨22478⟩⟩
def transferEvent : Nat := 76039
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 76037 .coefficient) (.value (.predecessor 1 76038 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76037 .coefficient)
      LeftAuthority76035.bound (LeftAuthority76035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76038 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76035.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76035.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76035.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76039

namespace LeftBound76043
def owner : Owner := ⟨.program ⟨214⟩, ⟨22479⟩⟩
def transferEvent : Nat := 76043
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76041 .coefficient) (.predecessor 1 76042 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76041 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76042 .coefficient)
      LeftBound76039.bound (LeftBound76039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76039.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound76039.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound76039.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound76039.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76043

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
