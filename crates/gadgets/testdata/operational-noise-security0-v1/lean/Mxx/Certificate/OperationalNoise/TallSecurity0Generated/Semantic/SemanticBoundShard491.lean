import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard083
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard084
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard437
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard440

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72057
def owner : Owner := ⟨.program ⟨214⟩, ⟨11130⟩⟩
def transferEvent : Nat := 72057
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 72055 .coefficient) (.predecessor 1 72056 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72055 .coefficient)
      LeftAuthority3407.bound (LeftAuthority3407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3408RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3407.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72056 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3407.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3407.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3407.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72057

namespace LeftBound72062
def owner : Owner := ⟨.program ⟨214⟩, ⟨7193⟩⟩
def transferEvent : Nat := 72062
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72060 .coefficient) (.predecessor 1 72061 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72060 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72061 .coefficient)
      LeftBound13485.bound (LeftBound13485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound13485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound13485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound13485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72062

namespace LeftBound72067
def owner : Owner := ⟨.program ⟨214⟩, ⟨11131⟩⟩
def transferEvent : Nat := 72067
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72065 .coefficient, .predecessor 1 72066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72065 .coefficient)
      LeftBound72062.bound (LeftBound72062.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72062.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72062.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72066 .coefficient)
      LeftBound72057.bound (LeftBound72057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72059RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72057.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72057.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72062.bound, LeftBound72057.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72062.bound, LeftBound72057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72062.actual selector witness, LeftBound72057.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72067

namespace LeftBound72071
def owner : Owner := ⟨.program ⟨214⟩, ⟨11132⟩⟩
def transferEvent : Nat := 72071
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72069 .coefficient, .predecessor 1 72070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72069 .coefficient)
      LeftBound72067.bound (LeftBound72067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72070 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72067.bound, LeftBound13477.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72067.bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72067.actual selector witness, LeftBound13477.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72071

namespace LeftBound72072
def owner : Owner := ⟨.program ⟨214⟩, ⟨11132⟩⟩
def transferEvent : Nat := 72072
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨89⟩⟩]⟩ [⟨.result 13478 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13478 .coefficient)
      LeftBound13477.bound (LeftBound13477.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨89⟩⟩) (rawTerms := some (Proof.Events052.exact13478RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13477.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13477.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13477.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72072

namespace LeftBound72077
def owner : Owner := ⟨.program ⟨214⟩, ⟨12157⟩⟩
def transferEvent : Nat := 72077
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72075 .coefficient) (.predecessor 1 72076 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72075 .coefficient)
      LeftBound72071.bound (LeftBound72071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72076 .coefficient)
      LeftAuthority3410.bound (LeftAuthority3410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound72071.bound LeftAuthority3410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72071.bound, LeftAuthority3410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound72071.actual selector witness) * (LeftAuthority3410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72077

namespace LeftBound72078
def owner : Owner := ⟨.program ⟨214⟩, ⟨12157⟩⟩
def transferEvent : Nat := 72078
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨12154⟩⟩], []⟩ [⟨.result 3411 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 3411 .coefficient)
      LeftAuthority3410.bound (LeftAuthority3410.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨12154⟩⟩) (rawTerms := some (Proof.Events013.exact3411RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3410.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority3410.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority3410.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72078

namespace LeftBound72079
def owner : Owner := ⟨.program ⟨214⟩, ⟨12157⟩⟩
def transferEvent : Nat := 72079
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72074 .summary) (.transfer 72078) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72074 .summary)
      LeftBound72072.bound (LeftBound72072.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨11132⟩⟩) (rawTerms := some (Proof.Events281.exact72074RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72072.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72078)
      LeftBound72078.bound (LeftBound72078.actual selector witness) := by
  exact .transfer (LeftBound72078.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound72072.bound LeftBound72078.bound
def bound : CoeffClass := .finite ⟨4992, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72072.bound, LeftBound72078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound72072.actual selector witness) * (LeftBound72078.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72079

namespace LeftBound72085
def owner : Owner := ⟨.program ⟨214⟩, ⟨12158⟩⟩
def transferEvent : Nat := 72085
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 72083 .coefficient) (.predecessor 1 72084 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72083 .coefficient)
      LeftAuthority3410.bound (LeftAuthority3410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3410.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72084 .coefficient)
      LeftBound65293.bound (LeftBound65293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65293.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65293.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority3410.bound LeftBound65293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3410.bound, LeftBound65293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority3410.actual selector witness) * (LeftBound65293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72085

namespace LeftBound72090
def owner : Owner := ⟨.program ⟨214⟩, ⟨7210⟩⟩
def transferEvent : Nat := 72090
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72088 .coefficient) (.predecessor 1 72089 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72088 .coefficient)
      LeftBound65164.bound (LeftBound65164.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65164.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72089 .coefficient)
      LeftBound13526.bound (LeftBound13526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13526.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound65164.bound LeftBound13526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65164.bound, LeftBound13526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound65164.actual selector witness) * (LeftBound13526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72090

namespace LeftBound72095
def owner : Owner := ⟨.program ⟨214⟩, ⟨12159⟩⟩
def transferEvent : Nat := 72095
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72093 .coefficient, .predecessor 1 72094 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72093 .coefficient)
      LeftBound72090.bound (LeftBound72090.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72090.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72094 .coefficient)
      LeftBound72085.bound (LeftBound72085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72090.bound, LeftBound72085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72090.bound, LeftBound72085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72090.actual selector witness, LeftBound72085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72095

namespace LeftBound72099
def owner : Owner := ⟨.program ⟨214⟩, ⟨12160⟩⟩
def transferEvent : Nat := 72099
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72097 .coefficient, .predecessor 1 72098 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72097 .coefficient)
      LeftBound72095.bound (LeftBound72095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72098 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72095.bound, LeftBound13518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72095.bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72095.actual selector witness, LeftBound13518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72099

namespace LeftBound72100
def owner : Owner := ⟨.program ⟨214⟩, ⟨12160⟩⟩
def transferEvent : Nat := 72100
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨106⟩⟩]⟩ [⟨.result 13519 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13519 .coefficient)
      LeftBound13518.bound (LeftBound13518.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨106⟩⟩) (rawTerms := some (Proof.Events052.exact13519RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13518.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound13518.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound13518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound13518.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72100

namespace LeftBound72105
def owner : Owner := ⟨.program ⟨214⟩, ⟨12161⟩⟩
def transferEvent : Nat := 72105
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72103 .coefficient) (.predecessor 1 72104 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72103 .coefficient)
      LeftBound72099.bound (LeftBound72099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events281.exact72102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72099.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72104 .coefficient)
      LeftBound13515.bound (LeftBound13515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound13515.bound, RecordedBoundRefines] <;> decide)
      (LeftBound13515.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72099.bound LeftBound13515.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72099.bound, LeftBound13515.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72099.actual selector witness) * (LeftBound13515.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72105

namespace LeftBound72106
def owner : Owner := ⟨.program ⟨214⟩, ⟨12161⟩⟩
def transferEvent : Nat := 72106
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨7840⟩⟩]⟩ [⟨.result 13512 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 13512 .coefficient)
      LeftAuthority13511.bound (LeftAuthority13511.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨7840⟩⟩) (rawTerms := some (Proof.Events052.exact13512RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13511.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13511.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority13511.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority13511.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72106

namespace LeftBound72107
def owner : Owner := ⟨.program ⟨214⟩, ⟨12161⟩⟩
def transferEvent : Nat := 72107
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72102 .summary) (.transfer 72106) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72102 .summary)
      LeftBound72100.bound (LeftBound72100.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12160⟩⟩) (rawTerms := some (Proof.Events281.exact72102RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72106)
      LeftBound72106.bound (LeftBound72106.actual selector witness) := by
  exact .transfer (LeftBound72106.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72100.bound LeftBound72106.bound
def bound : CoeffClass := .finite ⟨95420416, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72100.bound, LeftBound72106.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72100.actual selector witness) * (LeftBound72106.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72107

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
