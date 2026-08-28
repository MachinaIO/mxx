import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard666
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard667

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97661
def owner : Owner := ⟨.program ⟨214⟩, ⟨25131⟩⟩
def transferEvent : Nat := 97661
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97655 .summary, .result 97493 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97655 .summary)
      LeftBound97505.bound (LeftBound97505.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19736⟩⟩) (rawTerms := some (Proof.Events381.exact97655RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97505.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97493 .summary)
      LeftBound97488.bound (LeftBound97488.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25130⟩⟩) (rawTerms := some (Proof.Events380.exact97493RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97488.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97505.bound, LeftBound97488.bound]
def bound : CoeffClass := .finite ⟨352097360556032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97505.bound, LeftBound97488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97505.actual selector witness, LeftBound97488.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97661

namespace LeftBound97665
def owner : Owner := ⟨.program ⟨214⟩, ⟨28484⟩⟩
def transferEvent : Nat := 97665
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97663 .coefficient) (.predecessor 1 97664 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97663 .coefficient)
      LeftBound97658.bound (LeftBound97658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97662RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97658.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97664 .coefficient)
      LeftAuthority97408.bound (LeftAuthority97408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97658.bound LeftAuthority97408.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97658.bound, LeftAuthority97408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97658.actual selector witness) * (LeftAuthority97408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97665

namespace LeftBound97666
def owner : Owner := ⟨.program ⟨214⟩, ⟨28484⟩⟩
def transferEvent : Nat := 97666
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28482⟩⟩]⟩ [⟨.result 97409 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97409 .coefficient)
      LeftAuthority97408.bound (LeftAuthority97408.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28482⟩⟩) (rawTerms := some (Proof.Events380.exact97409RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97408.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97408.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97408.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97666

namespace LeftBound97667
def owner : Owner := ⟨.program ⟨214⟩, ⟨28484⟩⟩
def transferEvent : Nat := 97667
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 97662 .summary) (.transfer 97666) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97662 .summary)
      LeftBound97661.bound (LeftBound97661.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25131⟩⟩) (rawTerms := some (Proof.Events381.exact97662RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97661.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97666)
      LeftBound97666.bound (LeftBound97666.actual selector witness) := by
  exact .transfer (LeftBound97666.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97661.bound LeftBound97666.bound
def bound : CoeffClass := .finite ⟨1292202946798406336512, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97661.bound, LeftBound97666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97661.actual selector witness) * (LeftBound97666.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97667

namespace LeftBound97678
def owner : Owner := ⟨.program ⟨214⟩, ⟨21823⟩⟩
def transferEvent : Nat := 97678
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 97676 .coefficient) (.value (.predecessor 1 97677 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97676 .coefficient)
      LeftAuthority97674.bound (LeftAuthority97674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97674.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97677 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority97674.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97674.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97674.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97678

namespace LeftBound97682
def owner : Owner := ⟨.program ⟨214⟩, ⟨21824⟩⟩
def transferEvent : Nat := 97682
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97680 .coefficient) (.predecessor 1 97681 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97680 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97681 .coefficient)
      LeftBound97678.bound (LeftBound97678.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97679RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97678.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97678.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound97678.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound97678.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound97678.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97682

namespace LeftBound97683
def owner : Owner := ⟨.program ⟨214⟩, ⟨21824⟩⟩
def transferEvent : Nat := 97683
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21821⟩⟩]⟩ [⟨.result 97675 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 97675 .coefficient)
      LeftAuthority97674.bound (LeftAuthority97674.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21821⟩⟩) (rawTerms := some (Proof.Events381.exact97675RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97674.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority97674.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority97674.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97683

namespace LeftBound97684
def owner : Owner := ⟨.program ⟨214⟩, ⟨21824⟩⟩
def transferEvent : Nat := 97684
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 97683) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 97683)
      LeftBound97683.bound (LeftBound97683.actual selector witness) := by
  exact .transfer (LeftBound97683.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound97683.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound97683.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound97683.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97684

namespace LeftBound97755
def owner : Owner := ⟨.program ⟨214⟩, ⟨16253⟩⟩
def transferEvent : Nat := 97755
def frameStart : Nat := 97728
def rule : BoundRule := .identity (.predecessor 0 97754 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97754 .coefficient)
      LeftAuthority97752.bound (LeftAuthority97752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97752.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97752.derived selector witness)

def rawBound : CoeffClass := LeftAuthority97752.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97752.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority97752.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97755

namespace LeftBound97772
def owner : Owner := ⟨.program ⟨214⟩, ⟨16329⟩⟩
def transferEvent : Nat := 97772
def frameStart : Nat := 97728
def rule : BoundRule := .sum [.predecessor 0 97770 .coefficient, .predecessor 1 97771 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97770 .coefficient)
      LeftBound97755.bound (LeftBound97755.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97771 .coefficient)
      LeftAuthority97768.bound (LeftAuthority97768.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority97768.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97755.bound, LeftAuthority97768.bound]
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97755.bound, LeftAuthority97768.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound97755.actual selector witness, LeftAuthority97768.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97772

namespace LeftBound97775
def owner : Owner := ⟨.program ⟨214⟩, ⟨16330⟩⟩
def transferEvent : Nat := 97775
def frameStart : Nat := 97728
def rule : BoundRule := .identity (.predecessor 0 97774 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97774 .coefficient)
      LeftBound97772.bound (LeftBound97772.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97772.derived selector witness)

def rawBound : CoeffClass := LeftBound97772.bound
def bound : CoeffClass := .finite ⟨30, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97772.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound97772.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97775

namespace LeftBound97781
def owner : Owner := ⟨.program ⟨214⟩, ⟨16331⟩⟩
def transferEvent : Nat := 97781
def frameStart : Nat := 97728
def rule : BoundRule := .product (.predecessor 0 97779 .coefficient) (.predecessor 1 97780 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97779 .coefficient)
      LeftAuthority97777.bound (LeftAuthority97777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97780 .coefficient)
      LeftBound97775.bound (LeftBound97775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97775.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97775.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority97777.bound LeftBound97775.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97777.bound, LeftBound97775.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority97777.actual selector witness) * (LeftBound97775.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97781

namespace LeftBound97789
def owner : Owner := ⟨.program ⟨214⟩, ⟨16332⟩⟩
def transferEvent : Nat := 97789
def frameStart : Nat := 97728
def rule : BoundRule := .sum [.predecessor 0 97787 .coefficient, .predecessor 1 97788 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97787 .coefficient)
      LeftAuthority97785.bound (LeftAuthority97785.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97786RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97785.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97785.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97788 .coefficient)
      LeftBound97781.bound (LeftBound97781.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97781.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97781.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97785.bound, LeftBound97781.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97785.bound, LeftBound97781.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97785.actual selector witness, LeftBound97781.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97789

namespace LeftBound97793
def owner : Owner := ⟨.program ⟨214⟩, ⟨28483⟩⟩
def transferEvent : Nat := 97793
def frameStart : Nat := 97728
def rule : BoundRule := .product (.predecessor 0 97791 .coefficient) (.predecessor 1 97792 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97791 .coefficient)
      LeftBound97789.bound (LeftBound97789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97789.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97789.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97792 .coefficient)
      LeftAuthority97766.bound (LeftAuthority97766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97766.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound97789.bound LeftAuthority97766.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97789.bound, LeftAuthority97766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound97789.actual selector witness) * (LeftAuthority97766.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97793

namespace LeftBound97804
def owner : Owner := ⟨.program ⟨214⟩, ⟨16302⟩⟩
def transferEvent : Nat := 97804
def frameStart : Nat := 97728
def rule : BoundRule := .product (.predecessor 0 97802 .coefficient) (.predecessor 1 97803 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97802 .coefficient)
      LeftAuthority97777.bound (LeftAuthority97777.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events381.exact97778RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97777.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97777.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97803 .coefficient)
      LeftAuthority97800.bound (LeftAuthority97800.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97801RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97800.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97800.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97777.bound LeftAuthority97800.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97777.bound, LeftAuthority97800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority97777.actual selector witness) * (LeftAuthority97800.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97804

namespace LeftBound97812
def owner : Owner := ⟨.program ⟨214⟩, ⟨16303⟩⟩
def transferEvent : Nat := 97812
def frameStart : Nat := 97728
def rule : BoundRule := .sum [.predecessor 0 97810 .coefficient, .predecessor 1 97811 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 97810 .coefficient)
      LeftAuthority97808.bound (LeftAuthority97808.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97809RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97808.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97808.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 97811 .coefficient)
      LeftBound97804.bound (LeftBound97804.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events382.exact97806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97804.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97804.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97808.bound, LeftBound97804.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97808.bound, LeftBound97804.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority97808.actual selector witness, LeftBound97804.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97812

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
