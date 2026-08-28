import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard055
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard112

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18355
def owner : Owner := ⟨.program ⟨214⟩, ⟨29000⟩⟩
def transferEvent : Nat := 18355
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6669⟩⟩]⟩ [⟨.result 5615 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5615 .coefficient)
      LeftAuthority5614.bound (LeftAuthority5614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6669⟩⟩) (rawTerms := some (Proof.Events021.exact5615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5614.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18355

namespace LeftBound18356
def owner : Owner := ⟨.program ⟨214⟩, ⟨29000⟩⟩
def transferEvent : Nat := 18356
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 18351 .summary) (.transfer 18355) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18351 .summary)
      LeftBound18350.bound (LeftBound18350.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28999⟩⟩) (rawTerms := some (Proof.Events071.exact18351RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18350.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18355)
      LeftBound18355.bound (LeftBound18355.actual selector witness) := by
  exact .transfer (LeftBound18355.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18350.bound LeftBound18355.bound
def bound : CoeffClass := .finite ⟨4742816766803936246568583168, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18350.bound, LeftBound18355.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18350.actual selector witness) * (LeftBound18355.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18356

namespace LeftBound18371
def owner : Owner := ⟨.program ⟨214⟩, ⟨28781⟩⟩
def transferEvent : Nat := 18371
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18369 .coefficient) (.predecessor 1 18370 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18369 .coefficient)
      LeftBound9749.bound (LeftBound9749.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events038.exact9753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound9749.bound, RecordedBoundRefines] <;> decide)
      (LeftBound9749.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18370 .coefficient)
      LeftAuthority18367.bound (LeftAuthority18367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9749.bound LeftAuthority18367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9749.bound, LeftAuthority18367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9749.actual selector witness) * (LeftAuthority18367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18371

namespace LeftBound18372
def owner : Owner := ⟨.program ⟨214⟩, ⟨28781⟩⟩
def transferEvent : Nat := 18372
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28779⟩⟩]⟩ [⟨.result 18368 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18368 .coefficient)
      LeftAuthority18367.bound (LeftAuthority18367.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28779⟩⟩) (rawTerms := some (Proof.Events071.exact18368RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18367.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18367.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18367.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18372

namespace LeftBound18373
def owner : Owner := ⟨.program ⟨214⟩, ⟨28781⟩⟩
def transferEvent : Nat := 18373
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 9753 .summary) (.transfer 18372) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 9753 .summary)
      LeftBound9752.bound (LeftBound9752.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25241⟩⟩) (rawTerms := some (Proof.Events038.exact9753RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound9752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18372)
      LeftBound18372.bound (LeftBound18372.actual selector witness) := by
  exact .transfer (LeftBound18372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound9752.bound LeftBound18372.bound
def bound : CoeffClass := .finite ⟨1292270184133468094464, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound9752.bound, LeftBound18372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound9752.actual selector witness) * (LeftBound18372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18373

namespace LeftBound18384
def owner : Owner := ⟨.program ⟨214⟩, ⟨21922⟩⟩
def transferEvent : Nat := 18384
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 18382 .coefficient) (.value (.predecessor 1 18383 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18382 .coefficient)
      LeftAuthority18380.bound (LeftAuthority18380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18380.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18383 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority18380.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18380.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18380.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound18384

namespace LeftBound18388
def owner : Owner := ⟨.program ⟨214⟩, ⟨21923⟩⟩
def transferEvent : Nat := 18388
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 18386 .coefficient) (.predecessor 1 18387 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18386 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18387 .coefficient)
      LeftBound18384.bound (LeftBound18384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events071.exact18385RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18384.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound18384.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound18384.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound18384.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18388

namespace LeftBound18389
def owner : Owner := ⟨.program ⟨214⟩, ⟨21923⟩⟩
def transferEvent : Nat := 18389
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21920⟩⟩]⟩ [⟨.result 18381 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18381 .coefficient)
      LeftAuthority18380.bound (LeftAuthority18380.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21920⟩⟩) (rawTerms := some (Proof.Events071.exact18381RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18380.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18380.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority18380.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority18380.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound18389

namespace LeftBound18390
def owner : Owner := ⟨.program ⟨214⟩, ⟨21923⟩⟩
def transferEvent : Nat := 18390
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 18389) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18389)
      LeftBound18389.bound (LeftBound18389.actual selector witness) := by
  exact .transfer (LeftBound18389.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound18389.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound18389.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound18389.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18390

namespace LeftBound18485
def owner : Owner := ⟨.program ⟨214⟩, ⟨16398⟩⟩
def transferEvent : Nat := 18485
def frameStart : Nat := 18446
def rule : BoundRule := .identity (.predecessor 0 18484 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18484 .coefficient)
      LeftAuthority18482.bound (LeftAuthority18482.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18483RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18482.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18482.derived selector witness)

def rawBound : CoeffClass := LeftAuthority18482.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18482.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority18482.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18485

namespace LeftBound18502
def owner : Owner := ⟨.program ⟨214⟩, ⟨16437⟩⟩
def transferEvent : Nat := 18502
def frameStart : Nat := 18446
def rule : BoundRule := .sum [.predecessor 0 18500 .coefficient, .predecessor 1 18501 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18500 .coefficient)
      LeftBound18485.bound (LeftBound18485.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18485.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18501 .coefficient)
      LeftAuthority18498.bound (LeftAuthority18498.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority18498.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound18485.bound, LeftAuthority18498.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18485.bound, LeftAuthority18498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound18485.actual selector witness, LeftAuthority18498.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18502

namespace LeftBound18505
def owner : Owner := ⟨.program ⟨214⟩, ⟨16438⟩⟩
def transferEvent : Nat := 18505
def frameStart : Nat := 18446
def rule : BoundRule := .identity (.predecessor 0 18504 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18504 .coefficient)
      LeftBound18502.bound (LeftBound18502.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound18502.derived selector witness)

def rawBound : CoeffClass := LeftBound18502.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18502.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound18502.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound18505

namespace LeftBound18511
def owner : Owner := ⟨.program ⟨214⟩, ⟨16439⟩⟩
def transferEvent : Nat := 18511
def frameStart : Nat := 18446
def rule : BoundRule := .product (.predecessor 0 18509 .coefficient) (.predecessor 1 18510 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18509 .coefficient)
      LeftAuthority18507.bound (LeftAuthority18507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18510 .coefficient)
      LeftBound18505.bound (LeftBound18505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18506RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18505.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority18507.bound LeftBound18505.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18507.bound, LeftBound18505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority18507.actual selector witness) * (LeftBound18505.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18511

namespace LeftBound18519
def owner : Owner := ⟨.program ⟨214⟩, ⟨16440⟩⟩
def transferEvent : Nat := 18519
def frameStart : Nat := 18446
def rule : BoundRule := .sum [.predecessor 0 18517 .coefficient, .predecessor 1 18518 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18517 .coefficient)
      LeftAuthority18515.bound (LeftAuthority18515.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18516RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18515.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18515.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18518 .coefficient)
      LeftBound18511.bound (LeftBound18511.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18511.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18511.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority18515.bound, LeftBound18511.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18515.bound, LeftBound18511.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority18515.actual selector witness, LeftBound18511.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound18519

namespace LeftBound18523
def owner : Owner := ⟨.program ⟨214⟩, ⟨28780⟩⟩
def transferEvent : Nat := 18523
def frameStart : Nat := 18446
def rule : BoundRule := .product (.predecessor 0 18521 .coefficient) (.predecessor 1 18522 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18521 .coefficient)
      LeftBound18519.bound (LeftBound18519.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound18519.bound, RecordedBoundRefines] <;> decide)
      (LeftBound18519.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18522 .coefficient)
      LeftAuthority18496.bound (LeftAuthority18496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18496.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18519.bound LeftAuthority18496.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18519.bound, LeftAuthority18496.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18519.actual selector witness) * (LeftAuthority18496.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18523

namespace LeftBound18534
def owner : Owner := ⟨.program ⟨214⟩, ⟨18902⟩⟩
def transferEvent : Nat := 18534
def frameStart : Nat := 18446
def rule : BoundRule := .product (.predecessor 0 18532 .coefficient) (.predecessor 1 18533 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 18532 .coefficient)
      LeftAuthority18507.bound (LeftAuthority18507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18507.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 18533 .coefficient)
      LeftAuthority18530.bound (LeftAuthority18530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events072.exact18531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority18530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority18530.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority18507.bound LeftAuthority18530.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority18507.bound, LeftAuthority18530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority18507.actual selector witness) * (LeftAuthority18530.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18534

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
