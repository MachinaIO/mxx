import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard052
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard335
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard338
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard359

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound53495
def owner : Owner := ⟨.program ⟨214⟩, ⟨16510⟩⟩
def transferEvent : Nat := 53495
def frameStart : Nat := 53436
def rule : BoundRule := .identity (.predecessor 0 53494 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53494 .coefficient)
      LeftBound53492.bound (LeftBound53492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound53492.derived selector witness)

def rawBound : CoeffClass := LeftBound53492.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound53492.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound53495

namespace LeftBound53501
def owner : Owner := ⟨.program ⟨214⟩, ⟨16511⟩⟩
def transferEvent : Nat := 53501
def frameStart : Nat := 53436
def rule : BoundRule := .product (.predecessor 0 53499 .coefficient) (.predecessor 1 53500 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53499 .coefficient)
      LeftAuthority53497.bound (LeftAuthority53497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53500 .coefficient)
      LeftBound53495.bound (LeftBound53495.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53495.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53495.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority53497.bound LeftBound53495.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53497.bound, LeftBound53495.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority53497.actual selector witness) * (LeftBound53495.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53501

namespace LeftBound53509
def owner : Owner := ⟨.program ⟨214⟩, ⟨16512⟩⟩
def transferEvent : Nat := 53509
def frameStart : Nat := 53436
def rule : BoundRule := .sum [.predecessor 0 53507 .coefficient, .predecessor 1 53508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53507 .coefficient)
      LeftAuthority53505.bound (LeftAuthority53505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53505.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53508 .coefficient)
      LeftBound53501.bound (LeftBound53501.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53503RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53501.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53501.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53505.bound, LeftBound53501.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53505.bound, LeftBound53501.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53505.actual selector witness, LeftBound53501.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53509

namespace LeftBound53513
def owner : Owner := ⟨.program ⟨214⟩, ⟨28965⟩⟩
def transferEvent : Nat := 53513
def frameStart : Nat := 53436
def rule : BoundRule := .product (.predecessor 0 53511 .coefficient) (.predecessor 1 53512 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53511 .coefficient)
      LeftBound53509.bound (LeftBound53509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53510RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53512 .coefficient)
      LeftAuthority53486.bound (LeftAuthority53486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53486.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound53509.bound LeftAuthority53486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53509.bound, LeftAuthority53486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound53509.actual selector witness) * (LeftAuthority53486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53513

namespace LeftBound53524
def owner : Owner := ⟨.program ⟨214⟩, ⟨17908⟩⟩
def transferEvent : Nat := 53524
def frameStart : Nat := 53436
def rule : BoundRule := .product (.predecessor 0 53522 .coefficient) (.predecessor 1 53523 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53522 .coefficient)
      LeftAuthority53497.bound (LeftAuthority53497.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53498RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53497.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53497.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53523 .coefficient)
      LeftAuthority53520.bound (LeftAuthority53520.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53521RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53520.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53520.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority53497.bound LeftAuthority53520.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53497.bound, LeftAuthority53520.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority53497.actual selector witness) * (LeftAuthority53520.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53524

namespace LeftBound53532
def owner : Owner := ⟨.program ⟨214⟩, ⟨17909⟩⟩
def transferEvent : Nat := 53532
def frameStart : Nat := 53436
def rule : BoundRule := .sum [.predecessor 0 53530 .coefficient, .predecessor 1 53531 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53530 .coefficient)
      LeftAuthority53528.bound (LeftAuthority53528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53531 .coefficient)
      LeftBound53524.bound (LeftBound53524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53524.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53524.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority53528.bound, LeftBound53524.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53528.bound, LeftBound53524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority53528.actual selector witness, LeftBound53524.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53532

namespace LeftBound53536
def owner : Owner := ⟨.program ⟨214⟩, ⟨28969⟩⟩
def transferEvent : Nat := 53536
def frameStart : Nat := 53436
def rule : BoundRule := .sum [.predecessor 0 53534 .coefficient, .predecessor 1 53535 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53534 .coefficient)
      LeftBound53532.bound (LeftBound53532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53535 .coefficient)
      LeftBound53513.bound (LeftBound53513.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53513.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53513.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53532.bound, LeftBound53513.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53532.bound, LeftBound53513.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53532.actual selector witness, LeftBound53513.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53536

namespace LeftBound53549
def owner : Owner := ⟨.program ⟨214⟩, ⟨28967⟩⟩
def transferEvent : Nat := 53549
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53547 .coefficient, .predecessor 1 53548 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53547 .coefficient)
      LeftBound53378.bound (LeftBound53378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53378.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53548 .coefficient)
      LeftBound53361.bound (LeftBound53361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events208.exact53368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53361.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53361.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53378.bound, LeftBound53361.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53378.bound, LeftBound53361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53378.actual selector witness, LeftBound53361.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53549

namespace LeftBound53552
def owner : Owner := ⟨.program ⟨214⟩, ⟨28967⟩⟩
def transferEvent : Nat := 53552
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 53546 .summary, .result 53368 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53546 .summary)
      LeftBound53380.bound (LeftBound53380.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨22127⟩⟩) (rawTerms := some (Proof.Events209.exact53546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 53368 .summary)
      LeftBound53363.bound (LeftBound53363.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28966⟩⟩) (rawTerms := some (Proof.Events208.exact53368RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53363.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53380.bound, LeftBound53363.bound]
def bound : CoeffClass := .finite ⟨1292315010834812776448, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53380.bound, LeftBound53363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53380.actual selector witness, LeftBound53363.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53552

namespace LeftBound53576
def owner : Owner := ⟨.program ⟨214⟩, ⟨11968⟩⟩
def transferEvent : Nat := 53576
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 53574 .coefficient) (.predecessor 1 53575 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53574 .coefficient)
      LeftAuthority2475.bound (LeftAuthority2475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2475.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53575 .coefficient)
      LeftBound50668.bound (LeftBound50668.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50670RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50668.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50668.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority2475.bound LeftBound50668.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2475.bound, LeftBound50668.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority2475.actual selector witness) * (LeftBound50668.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound53576

namespace LeftBound53581
def owner : Owner := ⟨.program ⟨214⟩, ⟨7278⟩⟩
def transferEvent : Nat := 53581
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53579 .coefficient) (.predecessor 1 53580 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53579 .coefficient)
      LeftBound50539.bound (LeftBound50539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events197.exact50540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53580 .coefficient)
      LeftBound9477.bound (LeftBound9477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events037.exact9478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9477.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound50539.bound LeftBound9477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50539.bound, LeftBound9477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound50539.actual selector witness) * (LeftBound9477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 14) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53581

namespace LeftBound53586
def owner : Owner := ⟨.program ⟨214⟩, ⟨11969⟩⟩
def transferEvent : Nat := 53586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53584 .coefficient, .predecessor 1 53585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53584 .coefficient)
      LeftBound53581.bound (LeftBound53581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53581.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53585 .coefficient)
      LeftBound53576.bound (LeftBound53576.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53578RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53576.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53576.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53581.bound, LeftBound53576.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53581.bound, LeftBound53576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53581.actual selector witness, LeftBound53576.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53586

namespace LeftBound53590
def owner : Owner := ⟨.program ⟨214⟩, ⟨11970⟩⟩
def transferEvent : Nat := 53590
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 53588 .coefficient, .predecessor 1 53589 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53588 .coefficient)
      LeftBound53586.bound (LeftBound53586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53587RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53589 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53586.bound, LeftBound9469.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53586.bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound53586.actual selector witness, LeftBound9469.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound53590

namespace LeftBound53591
def owner : Owner := ⟨.program ⟨214⟩, ⟨11970⟩⟩
def transferEvent : Nat := 53591
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨98⟩⟩]⟩ [⟨.result 9470 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9470 .coefficient)
      LeftBound9469.bound (LeftBound9469.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨98⟩⟩) (rawTerms := some (Proof.Events036.exact9470RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9469.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9469.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound9469.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9469.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftBound9469.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53591

namespace LeftBound53596
def owner : Owner := ⟨.program ⟨214⟩, ⟨11971⟩⟩
def transferEvent : Nat := 53596
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 53594 .coefficient) (.predecessor 1 53595 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 53594 .coefficient)
      LeftBound53590.bound (LeftBound53590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events209.exact53593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 53595 .coefficient)
      LeftAuthority2478.bound (LeftAuthority2478.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2479RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2478.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound53590.bound LeftAuthority2478.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53590.bound, LeftAuthority2478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftBound53590.actual selector witness) * (LeftAuthority2478.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound53596

namespace LeftBound53597
def owner : Owner := ⟨.program ⟨214⟩, ⟨11971⟩⟩
def transferEvent : Nat := 53597
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨9720⟩⟩], []⟩ [⟨.result 2479 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 2479 .coefficient)
      LeftAuthority2478.bound (LeftAuthority2478.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨9720⟩⟩) (rawTerms := some (Proof.Events009.exact2479RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2478.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2478.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2478.bound []
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2478.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority2478.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound53597

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
