import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard422

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound63114
def owner : Owner := ⟨.program ⟨214⟩, ⟨21335⟩⟩
def transferEvent : Nat := 63114
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63112 .coefficient) (.predecessor 1 63113 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63112 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63113 .coefficient)
      LeftBound63110.bound (LeftBound63110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound63110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound63110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound63110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63114

namespace LeftBound63115
def owner : Owner := ⟨.program ⟨214⟩, ⟨21335⟩⟩
def transferEvent : Nat := 63115
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21332⟩⟩]⟩ [⟨.result 63107 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63107 .coefficient)
      LeftAuthority63106.bound (LeftAuthority63106.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21332⟩⟩) (rawTerms := some (Proof.Events246.exact63107RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63106.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63106.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority63106.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority63106.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63115

namespace LeftBound63116
def owner : Owner := ⟨.program ⟨214⟩, ⟨21335⟩⟩
def transferEvent : Nat := 63116
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 63115) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 63115)
      LeftBound63115.bound (LeftBound63115.actual selector witness) := by
  exact .transfer (LeftBound63115.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound63115.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound63115.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound63115.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63116

namespace LeftBound63211
def owner : Owner := ⟨.program ⟨214⟩, ⟨15945⟩⟩
def transferEvent : Nat := 63211
def frameStart : Nat := 63172
def rule : BoundRule := .identity (.predecessor 0 63210 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63210 .coefficient)
      LeftAuthority63208.bound (LeftAuthority63208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63208.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63208.derived selector witness)

def rawBound : CoeffClass := LeftAuthority63208.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority63208.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63211

namespace LeftBound63228
def owner : Owner := ⟨.program ⟨214⟩, ⟨16019⟩⟩
def transferEvent : Nat := 63228
def frameStart : Nat := 63172
def rule : BoundRule := .sum [.predecessor 0 63226 .coefficient, .predecessor 1 63227 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63226 .coefficient)
      LeftBound63211.bound (LeftBound63211.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63227 .coefficient)
      LeftAuthority63224.bound (LeftAuthority63224.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority63224.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63211.bound, LeftAuthority63224.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63211.bound, LeftAuthority63224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63211.actual selector witness, LeftAuthority63224.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63228

namespace LeftBound63231
def owner : Owner := ⟨.program ⟨214⟩, ⟨16020⟩⟩
def transferEvent : Nat := 63231
def frameStart : Nat := 63172
def rule : BoundRule := .identity (.predecessor 0 63230 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63230 .coefficient)
      LeftBound63228.bound (LeftBound63228.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound63228.derived selector witness)

def rawBound : CoeffClass := LeftBound63228.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound63228.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound63231

namespace LeftBound63237
def owner : Owner := ⟨.program ⟨214⟩, ⟨16021⟩⟩
def transferEvent : Nat := 63237
def frameStart : Nat := 63172
def rule : BoundRule := .product (.predecessor 0 63235 .coefficient) (.predecessor 1 63236 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63235 .coefficient)
      LeftAuthority63233.bound (LeftAuthority63233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63236 .coefficient)
      LeftBound63231.bound (LeftBound63231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63231.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority63233.bound LeftBound63231.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63233.bound, LeftBound63231.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority63233.actual selector witness) * (LeftBound63231.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63237

namespace LeftBound63245
def owner : Owner := ⟨.program ⟨214⟩, ⟨16022⟩⟩
def transferEvent : Nat := 63245
def frameStart : Nat := 63172
def rule : BoundRule := .sum [.predecessor 0 63243 .coefficient, .predecessor 1 63244 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63243 .coefficient)
      LeftAuthority63241.bound (LeftAuthority63241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63241.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63244 .coefficient)
      LeftBound63237.bound (LeftBound63237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63241.bound, LeftBound63237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63241.bound, LeftBound63237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63241.actual selector witness, LeftBound63237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63245

namespace LeftBound63249
def owner : Owner := ⟨.program ⟨214⟩, ⟨27873⟩⟩
def transferEvent : Nat := 63249
def frameStart : Nat := 63172
def rule : BoundRule := .product (.predecessor 0 63247 .coefficient) (.predecessor 1 63248 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63247 .coefficient)
      LeftBound63245.bound (LeftBound63245.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63246RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63245.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63245.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63248 .coefficient)
      LeftAuthority63222.bound (LeftAuthority63222.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63222.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63222.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63245.bound LeftAuthority63222.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63245.bound, LeftAuthority63222.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63245.actual selector witness) * (LeftAuthority63222.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63249

namespace LeftBound63260
def owner : Owner := ⟨.program ⟨214⟩, ⟨17171⟩⟩
def transferEvent : Nat := 63260
def frameStart : Nat := 63172
def rule : BoundRule := .product (.predecessor 0 63258 .coefficient) (.predecessor 1 63259 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63258 .coefficient)
      LeftAuthority63233.bound (LeftAuthority63233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63233.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63259 .coefficient)
      LeftAuthority63256.bound (LeftAuthority63256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority63233.bound LeftAuthority63256.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63233.bound, LeftAuthority63256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority63233.actual selector witness) * (LeftAuthority63256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63260

namespace LeftBound63268
def owner : Owner := ⟨.program ⟨214⟩, ⟨17172⟩⟩
def transferEvent : Nat := 63268
def frameStart : Nat := 63172
def rule : BoundRule := .sum [.predecessor 0 63266 .coefficient, .predecessor 1 63267 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63266 .coefficient)
      LeftAuthority63264.bound (LeftAuthority63264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority63264.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority63264.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63267 .coefficient)
      LeftBound63260.bound (LeftBound63260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority63264.bound, LeftBound63260.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority63264.bound, LeftBound63260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority63264.actual selector witness, LeftBound63260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63268

namespace LeftBound63272
def owner : Owner := ⟨.program ⟨214⟩, ⟨27878⟩⟩
def transferEvent : Nat := 63272
def frameStart : Nat := 63172
def rule : BoundRule := .sum [.predecessor 0 63270 .coefficient, .predecessor 1 63271 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63270 .coefficient)
      LeftBound63268.bound (LeftBound63268.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63269RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63268.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63268.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63271 .coefficient)
      LeftBound63249.bound (LeftBound63249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63249.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63249.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63268.bound, LeftBound63249.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63268.bound, LeftBound63249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63268.actual selector witness, LeftBound63249.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63272

namespace LeftBound63285
def owner : Owner := ⟨.program ⟨214⟩, ⟨27875⟩⟩
def transferEvent : Nat := 63285
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 63283 .coefficient, .predecessor 1 63284 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63283 .coefficient)
      LeftBound63114.bound (LeftBound63114.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63282RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63114.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63284 .coefficient)
      LeftBound63097.bound (LeftBound63097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events246.exact63104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63097.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63114.bound, LeftBound63097.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63114.bound, LeftBound63097.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63114.actual selector witness, LeftBound63097.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63285

namespace LeftBound63288
def owner : Owner := ⟨.program ⟨214⟩, ⟨27875⟩⟩
def transferEvent : Nat := 63288
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 63282 .summary, .result 63104 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63282 .summary)
      LeftBound63116.bound (LeftBound63116.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21335⟩⟩) (rawTerms := some (Proof.Events247.exact63282RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 63104 .summary)
      LeftBound63099.bound (LeftBound63099.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27874⟩⟩) (rawTerms := some (Proof.Events246.exact63104RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound63099.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound63116.bound, LeftBound63099.bound]
def bound : CoeffClass := .finite ⟨1292068473939586330624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63116.bound, LeftBound63099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound63116.actual selector witness, LeftBound63099.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound63288

namespace LeftBound63292
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def transferEvent : Nat := 63292
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 63290 .coefficient) (.predecessor 1 63291 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 63290 .coefficient)
      LeftBound63285.bound (LeftBound63285.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events247.exact63289RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound63285.bound, RecordedBoundRefines] <;> decide)
      (LeftBound63285.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 63291 .coefficient)
      LeftBound5718.bound (LeftBound5718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5718.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound63285.bound LeftBound5718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound63285.bound, LeftBound5718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound63285.actual selector witness) * (LeftBound5718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound63292

namespace LeftBound63293
def owner : Owner := ⟨.program ⟨214⟩, ⟨27876⟩⟩
def transferEvent : Nat := 63293
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6641⟩⟩]⟩ [⟨.result 5715 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5715 .coefficient)
      LeftAuthority5714.bound (LeftAuthority5714.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6641⟩⟩) (rawTerms := some (Proof.Events022.exact5715RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5714.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5714.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5714.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5714.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5714.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound63293

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
