import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard584
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard626

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound92524
def owner : Owner := ⟨.program ⟨214⟩, ⟨27644⟩⟩
def transferEvent : Nat := 92524
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27642⟩⟩]⟩ [⟨.result 92520 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92520 .coefficient)
      LeftAuthority92519.bound (LeftAuthority92519.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27642⟩⟩) (rawTerms := some (Proof.Events361.exact92520RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92519.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92519.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92519.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92519.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92524

namespace LeftBound92525
def owner : Owner := ⟨.program ⟨214⟩, ⟨27644⟩⟩
def transferEvent : Nat := 92525
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 85476 .summary) (.transfer 92524) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 85476 .summary)
      LeftBound85475.bound (LeftBound85475.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25991⟩⟩) (rawTerms := some (Proof.Events333.exact85476RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound85475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92524)
      LeftBound92524.bound (LeftBound92524.actual selector witness) := by
  exact .transfer (LeftBound92524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound85475.bound LeftBound92524.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85475.bound, LeftBound92524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound85475.actual selector witness) * (LeftBound92524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92525

namespace LeftBound92536
def owner : Owner := ⟨.program ⟨214⟩, ⟨21186⟩⟩
def transferEvent : Nat := 92536
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 92534 .coefficient) (.value (.predecessor 1 92535 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92534 .coefficient)
      LeftAuthority92532.bound (LeftAuthority92532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92535 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority92532.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92532.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92532.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound92536

namespace LeftBound92540
def owner : Owner := ⟨.program ⟨214⟩, ⟨21187⟩⟩
def transferEvent : Nat := 92540
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 92538 .coefficient) (.predecessor 1 92539 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92538 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92539 .coefficient)
      LeftBound92536.bound (LeftBound92536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92536.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound92536.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound92536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound92536.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92540

namespace LeftBound92541
def owner : Owner := ⟨.program ⟨214⟩, ⟨21187⟩⟩
def transferEvent : Nat := 92541
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21184⟩⟩]⟩ [⟨.result 92533 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 92533 .coefficient)
      LeftAuthority92532.bound (LeftAuthority92532.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21184⟩⟩) (rawTerms := some (Proof.Events361.exact92533RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92532.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority92532.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority92532.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound92541

namespace LeftBound92542
def owner : Owner := ⟨.program ⟨214⟩, ⟨21187⟩⟩
def transferEvent : Nat := 92542
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 92541) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 92541)
      LeftBound92541.bound (LeftBound92541.actual selector witness) := by
  exact .transfer (LeftBound92541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound92541.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound92541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound92541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92542

namespace LeftBound92637
def owner : Owner := ⟨.program ⟨214⟩, ⟨15822⟩⟩
def transferEvent : Nat := 92637
def frameStart : Nat := 92598
def rule : BoundRule := .identity (.predecessor 0 92636 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92636 .coefficient)
      LeftAuthority92634.bound (LeftAuthority92634.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92635RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92634.derived selector witness)

def rawBound : CoeffClass := LeftAuthority92634.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority92634.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92637

namespace LeftBound92654
def owner : Owner := ⟨.program ⟨214⟩, ⟨15896⟩⟩
def transferEvent : Nat := 92654
def frameStart : Nat := 92598
def rule : BoundRule := .sum [.predecessor 0 92652 .coefficient, .predecessor 1 92653 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92652 .coefficient)
      LeftBound92637.bound (LeftBound92637.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92637.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92653 .coefficient)
      LeftAuthority92650.bound (LeftAuthority92650.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority92650.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92637.bound, LeftAuthority92650.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92637.bound, LeftAuthority92650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92637.actual selector witness, LeftAuthority92650.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92654

namespace LeftBound92657
def owner : Owner := ⟨.program ⟨214⟩, ⟨15897⟩⟩
def transferEvent : Nat := 92657
def frameStart : Nat := 92598
def rule : BoundRule := .identity (.predecessor 0 92656 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92656 .coefficient)
      LeftBound92654.bound (LeftBound92654.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound92654.derived selector witness)

def rawBound : CoeffClass := LeftBound92654.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92654.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound92654.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound92657

namespace LeftBound92663
def owner : Owner := ⟨.program ⟨214⟩, ⟨15898⟩⟩
def transferEvent : Nat := 92663
def frameStart : Nat := 92598
def rule : BoundRule := .product (.predecessor 0 92661 .coefficient) (.predecessor 1 92662 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92661 .coefficient)
      LeftAuthority92659.bound (LeftAuthority92659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92662 .coefficient)
      LeftBound92657.bound (LeftBound92657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92658RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority92659.bound LeftBound92657.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92659.bound, LeftBound92657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority92659.actual selector witness) * (LeftBound92657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92663

namespace LeftBound92671
def owner : Owner := ⟨.program ⟨214⟩, ⟨15899⟩⟩
def transferEvent : Nat := 92671
def frameStart : Nat := 92598
def rule : BoundRule := .sum [.predecessor 0 92669 .coefficient, .predecessor 1 92670 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92669 .coefficient)
      LeftAuthority92667.bound (LeftAuthority92667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92667.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92670 .coefficient)
      LeftBound92663.bound (LeftBound92663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92663.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92663.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92667.bound, LeftBound92663.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92667.bound, LeftBound92663.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92667.actual selector witness, LeftBound92663.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92671

namespace LeftBound92675
def owner : Owner := ⟨.program ⟨214⟩, ⟨27643⟩⟩
def transferEvent : Nat := 92675
def frameStart : Nat := 92598
def rule : BoundRule := .product (.predecessor 0 92673 .coefficient) (.predecessor 1 92674 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92673 .coefficient)
      LeftBound92671.bound (LeftBound92671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92672RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92671.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92674 .coefficient)
      LeftAuthority92648.bound (LeftAuthority92648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92648.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound92671.bound LeftAuthority92648.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92671.bound, LeftAuthority92648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound92671.actual selector witness) * (LeftAuthority92648.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92675

namespace LeftBound92686
def owner : Owner := ⟨.program ⟨214⟩, ⟨17223⟩⟩
def transferEvent : Nat := 92686
def frameStart : Nat := 92598
def rule : BoundRule := .product (.predecessor 0 92684 .coefficient) (.predecessor 1 92685 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92684 .coefficient)
      LeftAuthority92659.bound (LeftAuthority92659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92660RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92659.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92659.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92685 .coefficient)
      LeftAuthority92682.bound (LeftAuthority92682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92682.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority92659.bound LeftAuthority92682.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92659.bound, LeftAuthority92682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority92659.actual selector witness) * (LeftAuthority92682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound92686

namespace LeftBound92694
def owner : Owner := ⟨.program ⟨214⟩, ⟨17224⟩⟩
def transferEvent : Nat := 92694
def frameStart : Nat := 92598
def rule : BoundRule := .sum [.predecessor 0 92692 .coefficient, .predecessor 1 92693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92692 .coefficient)
      LeftAuthority92690.bound (LeftAuthority92690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority92690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority92690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92693 .coefficient)
      LeftBound92686.bound (LeftBound92686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92686.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92686.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority92690.bound, LeftBound92686.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority92690.bound, LeftBound92686.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority92690.actual selector witness, LeftBound92686.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92694

namespace LeftBound92698
def owner : Owner := ⟨.program ⟨214⟩, ⟨27648⟩⟩
def transferEvent : Nat := 92698
def frameStart : Nat := 92598
def rule : BoundRule := .sum [.predecessor 0 92696 .coefficient, .predecessor 1 92697 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92696 .coefficient)
      LeftBound92694.bound (LeftBound92694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92695RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92697 .coefficient)
      LeftBound92675.bound (LeftBound92675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92675.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92675.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92694.bound, LeftBound92675.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92694.bound, LeftBound92675.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92694.actual selector witness, LeftBound92675.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92698

namespace LeftBound92711
def owner : Owner := ⟨.program ⟨214⟩, ⟨27645⟩⟩
def transferEvent : Nat := 92711
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 92709 .coefficient, .predecessor 1 92710 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 92709 .coefficient)
      LeftBound92540.bound (LeftBound92540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events362.exact92708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92540.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 92710 .coefficient)
      LeftBound92523.bound (LeftBound92523.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events361.exact92530RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound92523.bound, RecordedBoundRefines] <;> decide)
      (LeftBound92523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound92540.bound, LeftBound92523.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound92540.bound, LeftBound92523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound92540.actual selector witness, LeftBound92523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound92711

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
