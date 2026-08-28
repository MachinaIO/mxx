import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard553
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard554

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound81635
def owner : Owner := ⟨.program ⟨214⟩, ⟨25529⟩⟩
def transferEvent : Nat := 81635
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 81629 .summary, .result 81445 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81629 .summary)
      LeftBound81457.bound (LeftBound81457.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20035⟩⟩) (rawTerms := some (Proof.Events318.exact81629RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81445 .summary)
      LeftBound81440.bound (LeftBound81440.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25528⟩⟩) (rawTerms := some (Proof.Events318.exact81445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81440.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81457.bound, LeftBound81440.bound]
def bound : CoeffClass := .finite ⟨352146215809024, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81457.bound, LeftBound81440.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81457.actual selector witness, LeftBound81440.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81635

namespace LeftBound81639
def owner : Owner := ⟨.program ⟨214⟩, ⟨29387⟩⟩
def transferEvent : Nat := 81639
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81637 .coefficient) (.predecessor 1 81638 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81637 .coefficient)
      LeftBound81632.bound (LeftBound81632.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81636RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81632.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81638 .coefficient)
      LeftAuthority81360.bound (LeftAuthority81360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events317.exact81361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81632.bound LeftAuthority81360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81632.bound, LeftAuthority81360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81632.actual selector witness) * (LeftAuthority81360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81639

namespace LeftBound81640
def owner : Owner := ⟨.program ⟨214⟩, ⟨29387⟩⟩
def transferEvent : Nat := 81640
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29385⟩⟩]⟩ [⟨.result 81361 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81361 .coefficient)
      LeftAuthority81360.bound (LeftAuthority81360.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29385⟩⟩) (rawTerms := some (Proof.Events317.exact81361RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81360.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81360.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81360.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81360.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81640

namespace LeftBound81641
def owner : Owner := ⟨.program ⟨214⟩, ⟨29387⟩⟩
def transferEvent : Nat := 81641
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 81636 .summary) (.transfer 81640) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81636 .summary)
      LeftBound81635.bound (LeftBound81635.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25529⟩⟩) (rawTerms := some (Proof.Events318.exact81636RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound81635.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81640)
      LeftBound81640.bound (LeftBound81640.actual selector witness) := by
  exact .transfer (LeftBound81640.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81635.bound LeftBound81640.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81635.bound, LeftBound81640.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81635.actual selector witness) * (LeftBound81640.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81641

namespace LeftBound81652
def owner : Owner := ⟨.program ⟨214⟩, ⟨22410⟩⟩
def transferEvent : Nat := 81652
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 81650 .coefficient) (.value (.predecessor 1 81651 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81650 .coefficient)
      LeftAuthority81648.bound (LeftAuthority81648.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81649RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81648.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81651 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority81648.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81648.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81648.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound81652

namespace LeftBound81656
def owner : Owner := ⟨.program ⟨214⟩, ⟨22411⟩⟩
def transferEvent : Nat := 81656
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 81654 .coefficient) (.predecessor 1 81655 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81654 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81655 .coefficient)
      LeftBound81652.bound (LeftBound81652.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events318.exact81653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81652.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81652.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound81652.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound81652.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound81652.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81656

namespace LeftBound81657
def owner : Owner := ⟨.program ⟨214⟩, ⟨22411⟩⟩
def transferEvent : Nat := 81657
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22408⟩⟩]⟩ [⟨.result 81649 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 81649 .coefficient)
      LeftAuthority81648.bound (LeftAuthority81648.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22408⟩⟩) (rawTerms := some (Proof.Events318.exact81649RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81648.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81648.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority81648.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81648.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority81648.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound81657

namespace LeftBound81658
def owner : Owner := ⟨.program ⟨214⟩, ⟨22411⟩⟩
def transferEvent : Nat := 81658
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 81657) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 81657)
      LeftBound81657.bound (LeftBound81657.actual selector witness) := by
  exact .transfer (LeftBound81657.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound81657.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound81657.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound81657.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81658

namespace LeftBound81753
def owner : Owner := ⟨.program ⟨214⟩, ⟨16634⟩⟩
def transferEvent : Nat := 81753
def frameStart : Nat := 81714
def rule : BoundRule := .identity (.predecessor 0 81752 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81752 .coefficient)
      LeftAuthority81750.bound (LeftAuthority81750.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81750.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81750.derived selector witness)

def rawBound : CoeffClass := LeftAuthority81750.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81750.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority81750.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81753

namespace LeftBound81770
def owner : Owner := ⟨.program ⟨214⟩, ⟨16708⟩⟩
def transferEvent : Nat := 81770
def frameStart : Nat := 81714
def rule : BoundRule := .sum [.predecessor 0 81768 .coefficient, .predecessor 1 81769 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81768 .coefficient)
      LeftBound81753.bound (LeftBound81753.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81769 .coefficient)
      LeftAuthority81766.bound (LeftAuthority81766.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority81766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound81753.bound, LeftAuthority81766.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81753.bound, LeftAuthority81766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound81753.actual selector witness, LeftAuthority81766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81770

namespace LeftBound81773
def owner : Owner := ⟨.program ⟨214⟩, ⟨16709⟩⟩
def transferEvent : Nat := 81773
def frameStart : Nat := 81714
def rule : BoundRule := .identity (.predecessor 0 81772 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81772 .coefficient)
      LeftBound81770.bound (LeftBound81770.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound81770.derived selector witness)

def rawBound : CoeffClass := LeftBound81770.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81770.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound81770.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound81773

namespace LeftBound81779
def owner : Owner := ⟨.program ⟨214⟩, ⟨16710⟩⟩
def transferEvent : Nat := 81779
def frameStart : Nat := 81714
def rule : BoundRule := .product (.predecessor 0 81777 .coefficient) (.predecessor 1 81778 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81777 .coefficient)
      LeftAuthority81775.bound (LeftAuthority81775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81778 .coefficient)
      LeftBound81773.bound (LeftBound81773.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81774RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81773.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81773.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority81775.bound LeftBound81773.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81775.bound, LeftBound81773.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority81775.actual selector witness) * (LeftBound81773.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81779

namespace LeftBound81787
def owner : Owner := ⟨.program ⟨214⟩, ⟨16711⟩⟩
def transferEvent : Nat := 81787
def frameStart : Nat := 81714
def rule : BoundRule := .sum [.predecessor 0 81785 .coefficient, .predecessor 1 81786 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81785 .coefficient)
      LeftAuthority81783.bound (LeftAuthority81783.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81784RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81783.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81783.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81786 .coefficient)
      LeftBound81779.bound (LeftBound81779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81781RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81779.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81779.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81783.bound, LeftBound81779.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81783.bound, LeftBound81779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81783.actual selector witness, LeftBound81779.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81787

namespace LeftBound81791
def owner : Owner := ⟨.program ⟨214⟩, ⟨29386⟩⟩
def transferEvent : Nat := 81791
def frameStart : Nat := 81714
def rule : BoundRule := .product (.predecessor 0 81789 .coefficient) (.predecessor 1 81790 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81789 .coefficient)
      LeftBound81787.bound (LeftBound81787.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81788RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81787.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81787.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81790 .coefficient)
      LeftAuthority81764.bound (LeftAuthority81764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81764.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81764.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound81787.bound LeftAuthority81764.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound81787.bound, LeftAuthority81764.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound81787.actual selector witness) * (LeftAuthority81764.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81791

namespace LeftBound81802
def owner : Owner := ⟨.program ⟨214⟩, ⟨16680⟩⟩
def transferEvent : Nat := 81802
def frameStart : Nat := 81714
def rule : BoundRule := .product (.predecessor 0 81800 .coefficient) (.predecessor 1 81801 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81800 .coefficient)
      LeftAuthority81775.bound (LeftAuthority81775.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81776RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81775.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81775.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81801 .coefficient)
      LeftAuthority81798.bound (LeftAuthority81798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81799RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81798.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81798.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority81775.bound LeftAuthority81798.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81775.bound, LeftAuthority81798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority81775.actual selector witness) * (LeftAuthority81798.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound81802

namespace LeftBound81810
def owner : Owner := ⟨.program ⟨214⟩, ⟨16681⟩⟩
def transferEvent : Nat := 81810
def frameStart : Nat := 81714
def rule : BoundRule := .sum [.predecessor 0 81808 .coefficient, .predecessor 1 81809 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 81808 .coefficient)
      LeftAuthority81806.bound (LeftAuthority81806.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81807RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority81806.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority81806.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 81809 .coefficient)
      LeftBound81802.bound (LeftBound81802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events319.exact81804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound81802.bound, RecordedBoundRefines] <;> decide)
      (LeftBound81802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority81806.bound, LeftBound81802.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority81806.bound, LeftBound81802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority81806.actual selector witness, LeftBound81802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound81810

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
