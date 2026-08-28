import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard541
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard576

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound84532
def owner : Owner := ⟨.program ⟨214⟩, ⟨21546⟩⟩
def transferEvent : Nat := 84532
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 84530 .coefficient) (.value (.predecessor 1 84531 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84530 .coefficient)
      LeftAuthority84528.bound (LeftAuthority84528.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84529RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84528.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84531 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority84528.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84528.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84528.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84532

namespace LeftBound84536
def owner : Owner := ⟨.program ⟨214⟩, ⟨21547⟩⟩
def transferEvent : Nat := 84536
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 84534 .coefficient) (.predecessor 1 84535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84534 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84535 .coefficient)
      LeftBound84532.bound (LeftBound84532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84532.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84532.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound84532.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound84532.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound84532.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84536

namespace LeftBound84537
def owner : Owner := ⟨.program ⟨214⟩, ⟨21547⟩⟩
def transferEvent : Nat := 84537
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21544⟩⟩]⟩ [⟨.result 84529 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84529 .coefficient)
      LeftAuthority84528.bound (LeftAuthority84528.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21544⟩⟩) (rawTerms := some (Proof.Events330.exact84529RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84528.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84528.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority84528.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84528.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority84528.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound84537

namespace LeftBound84538
def owner : Owner := ⟨.program ⟨214⟩, ⟨21547⟩⟩
def transferEvent : Nat := 84538
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 84537) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 84537)
      LeftBound84537.bound (LeftBound84537.actual selector witness) := by
  exact .transfer (LeftBound84537.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound84537.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound84537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound84537.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84538

namespace LeftBound84633
def owner : Owner := ⟨.program ⟨214⟩, ⟨16060⟩⟩
def transferEvent : Nat := 84633
def frameStart : Nat := 84594
def rule : BoundRule := .identity (.predecessor 0 84632 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84632 .coefficient)
      LeftAuthority84630.bound (LeftAuthority84630.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84631RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84630.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84630.derived selector witness)

def rawBound : CoeffClass := LeftAuthority84630.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84630.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority84630.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84633

namespace LeftBound84650
def owner : Owner := ⟨.program ⟨214⟩, ⟨16134⟩⟩
def transferEvent : Nat := 84650
def frameStart : Nat := 84594
def rule : BoundRule := .sum [.predecessor 0 84648 .coefficient, .predecessor 1 84649 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84648 .coefficient)
      LeftBound84633.bound (LeftBound84633.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84633.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84649 .coefficient)
      LeftAuthority84646.bound (LeftAuthority84646.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority84646.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84633.bound, LeftAuthority84646.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84633.bound, LeftAuthority84646.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84633.actual selector witness, LeftAuthority84646.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84650

namespace LeftBound84653
def owner : Owner := ⟨.program ⟨214⟩, ⟨16135⟩⟩
def transferEvent : Nat := 84653
def frameStart : Nat := 84594
def rule : BoundRule := .identity (.predecessor 0 84652 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84652 .coefficient)
      LeftBound84650.bound (LeftBound84650.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound84650.derived selector witness)

def rawBound : CoeffClass := LeftBound84650.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound84650.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound84653

namespace LeftBound84659
def owner : Owner := ⟨.program ⟨214⟩, ⟨16136⟩⟩
def transferEvent : Nat := 84659
def frameStart : Nat := 84594
def rule : BoundRule := .product (.predecessor 0 84657 .coefficient) (.predecessor 1 84658 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84657 .coefficient)
      LeftAuthority84655.bound (LeftAuthority84655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84655.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84655.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84658 .coefficient)
      LeftBound84653.bound (LeftBound84653.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84654RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84653.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84653.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority84655.bound LeftBound84653.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84655.bound, LeftBound84653.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority84655.actual selector witness) * (LeftBound84653.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84659

namespace LeftBound84667
def owner : Owner := ⟨.program ⟨214⟩, ⟨16137⟩⟩
def transferEvent : Nat := 84667
def frameStart : Nat := 84594
def rule : BoundRule := .sum [.predecessor 0 84665 .coefficient, .predecessor 1 84666 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84665 .coefficient)
      LeftAuthority84663.bound (LeftAuthority84663.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84664RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84663.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84663.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84666 .coefficient)
      LeftBound84659.bound (LeftBound84659.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84659.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84659.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84663.bound, LeftBound84659.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84663.bound, LeftBound84659.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84663.actual selector witness, LeftBound84659.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84667

namespace LeftBound84671
def owner : Owner := ⟨.program ⟨214⟩, ⟨28084⟩⟩
def transferEvent : Nat := 84671
def frameStart : Nat := 84594
def rule : BoundRule := .product (.predecessor 0 84669 .coefficient) (.predecessor 1 84670 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84669 .coefficient)
      LeftBound84667.bound (LeftBound84667.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84668RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84667.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84667.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84670 .coefficient)
      LeftAuthority84644.bound (LeftAuthority84644.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84644.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84644.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound84667.bound LeftAuthority84644.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84667.bound, LeftAuthority84644.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound84667.actual selector witness) * (LeftAuthority84644.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84671

namespace LeftBound84682
def owner : Owner := ⟨.program ⟨214⟩, ⟨16106⟩⟩
def transferEvent : Nat := 84682
def frameStart : Nat := 84594
def rule : BoundRule := .product (.predecessor 0 84680 .coefficient) (.predecessor 1 84681 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84680 .coefficient)
      LeftAuthority84655.bound (LeftAuthority84655.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84656RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84655.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84655.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84681 .coefficient)
      LeftAuthority84678.bound (LeftAuthority84678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84678.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority84655.bound LeftAuthority84678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84655.bound, LeftAuthority84678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority84655.actual selector witness) * (LeftAuthority84678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound84682

namespace LeftBound84690
def owner : Owner := ⟨.program ⟨214⟩, ⟨16107⟩⟩
def transferEvent : Nat := 84690
def frameStart : Nat := 84594
def rule : BoundRule := .sum [.predecessor 0 84688 .coefficient, .predecessor 1 84689 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84688 .coefficient)
      LeftAuthority84686.bound (LeftAuthority84686.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84687RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority84686.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority84686.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84689 .coefficient)
      LeftBound84682.bound (LeftBound84682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84682.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84682.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority84686.bound, LeftBound84682.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority84686.bound, LeftBound84682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority84686.actual selector witness, LeftBound84682.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84690

namespace LeftBound84694
def owner : Owner := ⟨.program ⟨214⟩, ⟨28088⟩⟩
def transferEvent : Nat := 84694
def frameStart : Nat := 84594
def rule : BoundRule := .sum [.predecessor 0 84692 .coefficient, .predecessor 1 84693 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84692 .coefficient)
      LeftBound84690.bound (LeftBound84690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84690.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84690.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84693 .coefficient)
      LeftBound84671.bound (LeftBound84671.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84671.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84671.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84690.bound, LeftBound84671.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84690.bound, LeftBound84671.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84690.actual selector witness, LeftBound84671.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84694

namespace LeftBound84707
def owner : Owner := ⟨.program ⟨214⟩, ⟨28086⟩⟩
def transferEvent : Nat := 84707
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 84705 .coefficient, .predecessor 1 84706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84705 .coefficient)
      LeftBound84536.bound (LeftBound84536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84536.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84706 .coefficient)
      LeftBound84519.bound (LeftBound84519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84519.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84536.bound, LeftBound84519.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84536.bound, LeftBound84519.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84536.actual selector witness, LeftBound84519.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84707

namespace LeftBound84710
def owner : Owner := ⟨.program ⟨214⟩, ⟨28086⟩⟩
def transferEvent : Nat := 84710
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 84704 .summary, .result 84526 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84704 .summary)
      LeftBound84538.bound (LeftBound84538.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21547⟩⟩) (rawTerms := some (Proof.Events330.exact84704RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84538.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 84526 .summary)
      LeftBound84521.bound (LeftBound84521.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28085⟩⟩) (rawTerms := some (Proof.Events330.exact84526RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound84521.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84538.bound, LeftBound84521.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84538.bound, LeftBound84521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound84538.actual selector witness, LeftBound84521.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound84710

namespace LeftBound84734
def owner : Owner := ⟨.program ⟨214⟩, ⟨11470⟩⟩
def transferEvent : Nat := 84734
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 84732 .coefficient) (.predecessor 1 84733 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 84732 .coefficient)
      LeftAuthority4057.bound (LeftAuthority4057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events015.exact4058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4057.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 84733 .coefficient)
      LeftBound79918.bound (LeftBound79918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact79920RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound79918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound79918.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32 ⟨true, false, none, none, none⟩ LeftAuthority4057.bound LeftBound79918.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4057.bound, LeftBound79918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := tensorFactor 32 ⟨true, false, none, none, none⟩ * (LeftAuthority4057.actual selector witness) * (LeftBound79918.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound84734

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
