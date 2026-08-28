import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard037
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard447

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound66301
def owner : Owner := ⟨.program ⟨214⟩, ⟨10131⟩⟩
def transferEvent : Nat := 66301
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 66299 .coefficient) (.predecessor 1 66300 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66299 .coefficient)
      LeftAuthority3134.bound (LeftAuthority3134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events012.exact3135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3134.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66300 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3134.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3134.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3134.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66301

namespace LeftBound66306
def owner : Owner := ⟨.program ⟨214⟩, ⟨7186⟩⟩
def transferEvent : Nat := 66306
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66304 .coefficient) (.predecessor 1 66305 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66304 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66305 .coefficient)
      LeftBound7514.bound (LeftBound7514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7514.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound7514.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound7514.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound7514.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66306

namespace LeftBound66311
def owner : Owner := ⟨.program ⟨214⟩, ⟨10132⟩⟩
def transferEvent : Nat := 66311
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66309 .coefficient, .predecessor 1 66310 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66309 .coefficient)
      LeftBound66306.bound (LeftBound66306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66306.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66310 .coefficient)
      LeftBound66301.bound (LeftBound66301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66301.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66306.bound, LeftBound66301.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66306.bound, LeftBound66301.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66306.actual selector witness, LeftBound66301.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66311

namespace LeftBound66315
def owner : Owner := ⟨.program ⟨214⟩, ⟨10133⟩⟩
def transferEvent : Nat := 66315
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66313 .coefficient, .predecessor 1 66314 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66313 .coefficient)
      LeftBound66311.bound (LeftBound66311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66312RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66314 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66311.bound, LeftBound7506.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66311.bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66311.actual selector witness, LeftBound7506.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66315

namespace LeftBound66316
def owner : Owner := ⟨.program ⟨214⟩, ⟨10133⟩⟩
def transferEvent : Nat := 66316
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨82⟩⟩]⟩ [⟨.result 7507 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7507 .coefficient)
      LeftBound7506.bound (LeftBound7506.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨82⟩⟩) (rawTerms := some (Proof.Events029.exact7507RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7506.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7506.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound7506.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound7506.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound7506.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66316

namespace LeftBound66321
def owner : Owner := ⟨.program ⟨214⟩, ⟨10134⟩⟩
def transferEvent : Nat := 66321
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66319 .coefficient) (.predecessor 1 66320 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66319 .coefficient)
      LeftBound66315.bound (LeftBound66315.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66318RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66315.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66315.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66320 .coefficient)
      LeftBound7503.bound (LeftBound7503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound7503.bound, RecordedBoundRefines] <;> decide)
      (LeftBound7503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66315.bound LeftBound7503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66315.bound, LeftBound7503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66315.actual selector witness) * (LeftBound7503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66321

namespace LeftBound66322
def owner : Owner := ⟨.program ⟨214⟩, ⟨10134⟩⟩
def transferEvent : Nat := 66322
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7876⟩⟩]⟩ [⟨.result 7500 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7500 .coefficient)
      LeftAuthority7499.bound (LeftAuthority7499.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7876⟩⟩) (rawTerms := some (Proof.Events029.exact7500RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7499.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7499.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7499.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7499.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7499.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66322

namespace LeftBound66323
def owner : Owner := ⟨.program ⟨214⟩, ⟨10134⟩⟩
def transferEvent : Nat := 66323
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66318 .summary) (.transfer 66322) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66318 .summary)
      LeftBound66316.bound (LeftBound66316.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10133⟩⟩) (rawTerms := some (Proof.Events259.exact66318RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66322)
      LeftBound66322.bound (LeftBound66322.actual selector witness) := by
  exact .transfer (LeftBound66322.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66316.bound LeftBound66322.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66316.bound, LeftBound66322.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66316.actual selector witness) * (LeftBound66322.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66323

namespace LeftBound66331
def owner : Owner := ⟨.program ⟨214⟩, ⟨12957⟩⟩
def transferEvent : Nat := 66331
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 66329 .coefficient, .predecessor 1 66330 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66329 .coefficient)
      LeftBound66321.bound (LeftBound66321.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66321.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66321.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66330 .coefficient)
      LeftBound66293.bound (LeftBound66293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66298RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66293.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66321.bound, LeftBound66293.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66321.bound, LeftBound66293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66321.actual selector witness, LeftBound66293.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66331

namespace LeftBound66333
def owner : Owner := ⟨.program ⟨214⟩, ⟨12957⟩⟩
def transferEvent : Nat := 66333
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 66328 .summary, .result 66298 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66328 .summary)
      LeftBound66323.bound (LeftBound66323.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10134⟩⟩) (rawTerms := some (Proof.Events259.exact66328RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66323.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66298 .summary)
      LeftBound66295.bound (LeftBound66295.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12956⟩⟩) (rawTerms := some (Proof.Events258.exact66298RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66295.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound66323.bound, LeftBound66295.bound]
def bound : CoeffClass := .finite ⟨95463680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66323.bound, LeftBound66295.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound66323.actual selector witness, LeftBound66295.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound66333

namespace LeftBound66337
def owner : Owner := ⟨.program ⟨214⟩, ⟨25600⟩⟩
def transferEvent : Nat := 66337
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66335 .coefficient) (.predecessor 1 66336 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66335 .coefficient)
      LeftBound66331.bound (LeftBound66331.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66334RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66331.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66336 .coefficient)
      LeftAuthority66269.bound (LeftAuthority66269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events258.exact66270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66331.bound LeftAuthority66269.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66331.bound, LeftAuthority66269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66331.actual selector witness) * (LeftAuthority66269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66337

namespace LeftBound66338
def owner : Owner := ⟨.program ⟨214⟩, ⟨25600⟩⟩
def transferEvent : Nat := 66338
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25599⟩⟩]⟩ [⟨.result 66270 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66270 .coefficient)
      LeftAuthority66269.bound (LeftAuthority66269.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25599⟩⟩) (rawTerms := some (Proof.Events258.exact66270RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66269.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66269.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66269.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66338

namespace LeftBound66339
def owner : Owner := ⟨.program ⟨214⟩, ⟨25600⟩⟩
def transferEvent : Nat := 66339
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 66334 .summary) (.transfer 66338) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66334 .summary)
      LeftBound66333.bound (LeftBound66333.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12957⟩⟩) (rawTerms := some (Proof.Events259.exact66334RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound66333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 66338)
      LeftBound66338.bound (LeftBound66338.actual selector witness) := by
  exact .transfer (LeftBound66338.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound66333.bound LeftBound66338.bound
def bound : CoeffClass := .finite ⟨350353233018880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound66333.bound, LeftBound66338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound66333.actual selector witness) * (LeftBound66338.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66339

namespace LeftBound66350
def owner : Owner := ⟨.program ⟨214⟩, ⟨20102⟩⟩
def transferEvent : Nat := 66350
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 66348 .coefficient) (.value (.predecessor 1 66349 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66348 .coefficient)
      LeftAuthority66346.bound (LeftAuthority66346.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66347RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66346.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66346.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66349 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority66346.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66346.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66346.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound66350

namespace LeftBound66354
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def transferEvent : Nat := 66354
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 66352 .coefficient) (.predecessor 1 66353 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 66352 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 66353 .coefficient)
      LeftBound66350.bound (LeftBound66350.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events259.exact66351RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound66350.bound, RecordedBoundRefines] <;> decide)
      (LeftBound66350.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound66350.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound66350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound66350.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound66354

namespace LeftBound66355
def owner : Owner := ⟨.program ⟨214⟩, ⟨20103⟩⟩
def transferEvent : Nat := 66355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20100⟩⟩]⟩ [⟨.result 66347 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 66347 .coefficient)
      LeftAuthority66346.bound (LeftAuthority66346.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20100⟩⟩) (rawTerms := some (Proof.Events259.exact66347RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority66346.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority66346.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority66346.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority66346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority66346.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound66355

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
