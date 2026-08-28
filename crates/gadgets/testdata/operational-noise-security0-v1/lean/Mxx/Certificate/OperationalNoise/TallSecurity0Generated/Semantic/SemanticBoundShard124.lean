import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard020
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard090
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard123

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound20086
def owner : Owner := ⟨.program ⟨214⟩, ⟨20771⟩⟩
def transferEvent : Nat := 20086
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 20085) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20085)
      LeftBound20085.bound (LeftBound20085.actual selector witness) := by
  exact .transfer (LeftBound20085.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound20085.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound20085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound20085.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20086

namespace LeftBound20181
def owner : Owner := ⟨.program ⟨214⟩, ⟨15439⟩⟩
def transferEvent : Nat := 20181
def frameStart : Nat := 20142
def rule : BoundRule := .identity (.predecessor 0 20180 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20180 .coefficient)
      LeftAuthority20178.bound (LeftAuthority20178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20178.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20178.derived selector witness)

def rawBound : CoeffClass := LeftAuthority20178.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority20178.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20181

namespace LeftBound20198
def owner : Owner := ⟨.program ⟨214⟩, ⟨15478⟩⟩
def transferEvent : Nat := 20198
def frameStart : Nat := 20142
def rule : BoundRule := .sum [.predecessor 0 20196 .coefficient, .predecessor 1 20197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20196 .coefficient)
      LeftBound20181.bound (LeftBound20181.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20197 .coefficient)
      LeftAuthority20194.bound (LeftAuthority20194.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority20194.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20181.bound, LeftAuthority20194.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20181.bound, LeftAuthority20194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20181.actual selector witness, LeftAuthority20194.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20198

namespace LeftBound20201
def owner : Owner := ⟨.program ⟨214⟩, ⟨15479⟩⟩
def transferEvent : Nat := 20201
def frameStart : Nat := 20142
def rule : BoundRule := .identity (.predecessor 0 20200 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20200 .coefficient)
      LeftBound20198.bound (LeftBound20198.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound20198.derived selector witness)

def rawBound : CoeffClass := LeftBound20198.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound20198.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound20201

namespace LeftBound20207
def owner : Owner := ⟨.program ⟨214⟩, ⟨15480⟩⟩
def transferEvent : Nat := 20207
def frameStart : Nat := 20142
def rule : BoundRule := .product (.predecessor 0 20205 .coefficient) (.predecessor 1 20206 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20205 .coefficient)
      LeftAuthority20203.bound (LeftAuthority20203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20206 .coefficient)
      LeftBound20201.bound (LeftBound20201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20202RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20201.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority20203.bound LeftBound20201.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20203.bound, LeftBound20201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority20203.actual selector witness) * (LeftBound20201.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20207

namespace LeftBound20215
def owner : Owner := ⟨.program ⟨214⟩, ⟨15481⟩⟩
def transferEvent : Nat := 20215
def frameStart : Nat := 20142
def rule : BoundRule := .sum [.predecessor 0 20213 .coefficient, .predecessor 1 20214 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20213 .coefficient)
      LeftAuthority20211.bound (LeftAuthority20211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20214 .coefficient)
      LeftBound20207.bound (LeftBound20207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20207.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20211.bound, LeftBound20207.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20211.bound, LeftBound20207.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20211.actual selector witness, LeftBound20207.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20215

namespace LeftBound20219
def owner : Owner := ⟨.program ⟨214⟩, ⟨27044⟩⟩
def transferEvent : Nat := 20219
def frameStart : Nat := 20142
def rule : BoundRule := .product (.predecessor 0 20217 .coefficient) (.predecessor 1 20218 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20217 .coefficient)
      LeftBound20215.bound (LeftBound20215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20215.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20218 .coefficient)
      LeftAuthority20192.bound (LeftAuthority20192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20192.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20215.bound LeftAuthority20192.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20215.bound, LeftAuthority20192.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20215.actual selector witness) * (LeftAuthority20192.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20219

namespace LeftBound20230
def owner : Owner := ⟨.program ⟨214⟩, ⟨15539⟩⟩
def transferEvent : Nat := 20230
def frameStart : Nat := 20142
def rule : BoundRule := .product (.predecessor 0 20228 .coefficient) (.predecessor 1 20229 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20228 .coefficient)
      LeftAuthority20203.bound (LeftAuthority20203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20203.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20229 .coefficient)
      LeftAuthority20226.bound (LeftAuthority20226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20226.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority20203.bound LeftAuthority20226.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20203.bound, LeftAuthority20226.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority20203.actual selector witness) * (LeftAuthority20226.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20230

namespace LeftBound20238
def owner : Owner := ⟨.program ⟨214⟩, ⟨15540⟩⟩
def transferEvent : Nat := 20238
def frameStart : Nat := 20142
def rule : BoundRule := .sum [.predecessor 0 20236 .coefficient, .predecessor 1 20237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20236 .coefficient)
      LeftAuthority20234.bound (LeftAuthority20234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20237 .coefficient)
      LeftBound20230.bound (LeftBound20230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority20234.bound, LeftBound20230.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority20234.bound, LeftBound20230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority20234.actual selector witness, LeftBound20230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20238

namespace LeftBound20242
def owner : Owner := ⟨.program ⟨214⟩, ⟨27049⟩⟩
def transferEvent : Nat := 20242
def frameStart : Nat := 20142
def rule : BoundRule := .sum [.predecessor 0 20240 .coefficient, .predecessor 1 20241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20240 .coefficient)
      LeftBound20238.bound (LeftBound20238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20238.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20241 .coefficient)
      LeftBound20219.bound (LeftBound20219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20219.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20219.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20238.bound, LeftBound20219.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20238.bound, LeftBound20219.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20238.actual selector witness, LeftBound20219.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20242

namespace LeftBound20255
def owner : Owner := ⟨.program ⟨214⟩, ⟨27046⟩⟩
def transferEvent : Nat := 20255
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 20253 .coefficient, .predecessor 1 20254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20253 .coefficient)
      LeftBound20084.bound (LeftBound20084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20084.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20254 .coefficient)
      LeftBound20067.bound (LeftBound20067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events078.exact20074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20067.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20084.bound, LeftBound20067.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20084.bound, LeftBound20067.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20084.actual selector witness, LeftBound20067.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20255

namespace LeftBound20258
def owner : Owner := ⟨.program ⟨214⟩, ⟨27046⟩⟩
def transferEvent : Nat := 20258
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 20252 .summary, .result 20074 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20252 .summary)
      LeftBound20086.bound (LeftBound20086.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20771⟩⟩) (rawTerms := some (Proof.Events079.exact20252RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20074 .summary)
      LeftBound20069.bound (LeftBound20069.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27045⟩⟩) (rawTerms := some (Proof.Events078.exact20074RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20069.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound20086.bound, LeftBound20069.bound]
def bound : CoeffClass := .finite ⟨1291933999269462814720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20086.bound, LeftBound20069.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound20086.actual selector witness, LeftBound20069.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound20258

namespace LeftBound20262
def owner : Owner := ⟨.program ⟨214⟩, ⟨27047⟩⟩
def transferEvent : Nat := 20262
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20260 .coefficient) (.predecessor 1 20261 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20260 .coefficient)
      LeftBound20255.bound (LeftBound20255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound20255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound20255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20261 .coefficient)
      LeftBound5798.bound (LeftBound5798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20255.bound LeftBound5798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20255.bound, LeftBound5798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20255.actual selector witness) * (LeftBound5798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20262

namespace LeftBound20263
def owner : Owner := ⟨.program ⟨214⟩, ⟨27047⟩⟩
def transferEvent : Nat := 20263
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6655⟩⟩]⟩ [⟨.result 5795 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5795 .coefficient)
      LeftAuthority5794.bound (LeftAuthority5794.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6655⟩⟩) (rawTerms := some (Proof.Events022.exact5795RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5794.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5794.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5794.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5794.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5794.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound20263

namespace LeftBound20264
def owner : Owner := ⟨.program ⟨214⟩, ⟨27047⟩⟩
def transferEvent : Nat := 20264
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 20259 .summary) (.transfer 20263) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 20259 .summary)
      LeftBound20258.bound (LeftBound20258.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨27046⟩⟩) (rawTerms := some (Proof.Events079.exact20259RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound20258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 20263)
      LeftBound20263.bound (LeftBound20263.actual selector witness) := by
  exact .transfer (LeftBound20263.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound20258.bound LeftBound20263.bound
def bound : CoeffClass := .finite ⟨4741418448262916841427435520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound20258.bound, LeftBound20263.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound20258.actual selector witness) * (LeftBound20263.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20264

namespace LeftBound20279
def owner : Owner := ⟨.program ⟨214⟩, ⟨26828⟩⟩
def transferEvent : Nat := 20279
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 20277 .coefficient) (.predecessor 1 20278 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 20277 .coefficient)
      LeftBound14258.bound (LeftBound14258.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events055.exact14262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound14258.bound, RecordedBoundRefines] <;> decide)
      (LeftBound14258.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 20278 .coefficient)
      LeftAuthority20275.bound (LeftAuthority20275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events079.exact20276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority20275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority20275.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound14258.bound LeftAuthority20275.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound14258.bound, LeftAuthority20275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound14258.actual selector witness) * (LeftAuthority20275.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound20279

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
