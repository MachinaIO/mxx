import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard074

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12262
def owner : Owner := ⟨.program ⟨214⟩, ⟨27703⟩⟩
def transferEvent : Nat := 12262
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27701⟩⟩]⟩ [⟨.result 11962 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11962 .coefficient)
      LeftAuthority11961.bound (LeftAuthority11961.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27701⟩⟩) (rawTerms := some (Proof.Events046.exact11962RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority11961.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority11961.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority11961.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority11961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority11961.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12262

namespace LeftBound12263
def owner : Owner := ⟨.program ⟨214⟩, ⟨27703⟩⟩
def transferEvent : Nat := 12263
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 12258 .summary) (.transfer 12262) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12258 .summary)
      LeftBound12257.bound (LeftBound12257.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26011⟩⟩) (rawTerms := some (Proof.Events047.exact12258RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound12257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12262)
      LeftBound12262.bound (LeftBound12262.actual selector witness) := by
  exact .transfer (LeftBound12262.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12257.bound LeftBound12262.bound
def bound : CoeffClass := .finite ⟨1292046059683262234624, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12257.bound, LeftBound12262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12257.actual selector witness) * (LeftBound12262.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12263

namespace LeftBound12274
def owner : Owner := ⟨.program ⟨214⟩, ⟨21274⟩⟩
def transferEvent : Nat := 12274
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 12272 .coefficient) (.value (.predecessor 1 12273 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12272 .coefficient)
      LeftAuthority12270.bound (LeftAuthority12270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12273 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority12270.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12270.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12270.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound12274

namespace LeftBound12278
def owner : Owner := ⟨.program ⟨214⟩, ⟨21275⟩⟩
def transferEvent : Nat := 12278
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12276 .coefficient) (.predecessor 1 12277 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12276 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12277 .coefficient)
      LeftBound12274.bound (LeftBound12274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12274.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound12274.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound12274.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound12274.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12278

namespace LeftBound12279
def owner : Owner := ⟨.program ⟨214⟩, ⟨21275⟩⟩
def transferEvent : Nat := 12279
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21272⟩⟩]⟩ [⟨.result 12271 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 12271 .coefficient)
      LeftAuthority12270.bound (LeftAuthority12270.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21272⟩⟩) (rawTerms := some (Proof.Events047.exact12271RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12270.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12270.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority12270.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority12270.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound12279

namespace LeftBound12280
def owner : Owner := ⟨.program ⟨214⟩, ⟨21275⟩⟩
def transferEvent : Nat := 12280
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 12279) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 12279)
      LeftBound12279.bound (LeftBound12279.actual selector witness) := by
  exact .transfer (LeftBound12279.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound12279.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound12279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound12279.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12280

namespace LeftBound12375
def owner : Owner := ⟨.program ⟨214⟩, ⟨15838⟩⟩
def transferEvent : Nat := 12375
def frameStart : Nat := 12336
def rule : BoundRule := .identity (.predecessor 0 12374 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12374 .coefficient)
      LeftAuthority12372.bound (LeftAuthority12372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12372.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12372.derived selector witness)

def rawBound : CoeffClass := LeftAuthority12372.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority12372.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12375

namespace LeftBound12392
def owner : Owner := ⟨.program ⟨214⟩, ⟨15912⟩⟩
def transferEvent : Nat := 12392
def frameStart : Nat := 12336
def rule : BoundRule := .sum [.predecessor 0 12390 .coefficient, .predecessor 1 12391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12390 .coefficient)
      LeftBound12375.bound (LeftBound12375.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12375.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12391 .coefficient)
      LeftAuthority12388.bound (LeftAuthority12388.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority12388.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12375.bound, LeftAuthority12388.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12375.bound, LeftAuthority12388.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12375.actual selector witness, LeftAuthority12388.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12392

namespace LeftBound12395
def owner : Owner := ⟨.program ⟨214⟩, ⟨15913⟩⟩
def transferEvent : Nat := 12395
def frameStart : Nat := 12336
def rule : BoundRule := .identity (.predecessor 0 12394 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12394 .coefficient)
      LeftBound12392.bound (LeftBound12392.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound12392.derived selector witness)

def rawBound : CoeffClass := LeftBound12392.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound12392.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound12395

namespace LeftBound12401
def owner : Owner := ⟨.program ⟨214⟩, ⟨15914⟩⟩
def transferEvent : Nat := 12401
def frameStart : Nat := 12336
def rule : BoundRule := .product (.predecessor 0 12399 .coefficient) (.predecessor 1 12400 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12399 .coefficient)
      LeftAuthority12397.bound (LeftAuthority12397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12400 .coefficient)
      LeftBound12395.bound (LeftBound12395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12395.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12395.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority12397.bound LeftBound12395.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12397.bound, LeftBound12395.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority12397.actual selector witness) * (LeftBound12395.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12401

namespace LeftBound12409
def owner : Owner := ⟨.program ⟨214⟩, ⟨15915⟩⟩
def transferEvent : Nat := 12409
def frameStart : Nat := 12336
def rule : BoundRule := .sum [.predecessor 0 12407 .coefficient, .predecessor 1 12408 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12407 .coefficient)
      LeftAuthority12405.bound (LeftAuthority12405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12408 .coefficient)
      LeftBound12401.bound (LeftBound12401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12405.bound, LeftBound12401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12405.bound, LeftBound12401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12405.actual selector witness, LeftBound12401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12409

namespace LeftBound12413
def owner : Owner := ⟨.program ⟨214⟩, ⟨27702⟩⟩
def transferEvent : Nat := 12413
def frameStart : Nat := 12336
def rule : BoundRule := .product (.predecessor 0 12411 .coefficient) (.predecessor 1 12412 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12411 .coefficient)
      LeftBound12409.bound (LeftBound12409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12412 .coefficient)
      LeftAuthority12386.bound (LeftAuthority12386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12386.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12386.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound12409.bound LeftAuthority12386.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12409.bound, LeftAuthority12386.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound12409.actual selector witness) * (LeftAuthority12386.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12413

namespace LeftBound12424
def owner : Owner := ⟨.program ⟨214⟩, ⟨15880⟩⟩
def transferEvent : Nat := 12424
def frameStart : Nat := 12336
def rule : BoundRule := .product (.predecessor 0 12422 .coefficient) (.predecessor 1 12423 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12422 .coefficient)
      LeftAuthority12397.bound (LeftAuthority12397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12423 .coefficient)
      LeftAuthority12420.bound (LeftAuthority12420.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12421RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12420.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12420.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority12397.bound LeftAuthority12420.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12397.bound, LeftAuthority12420.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority12397.actual selector witness) * (LeftAuthority12420.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12424

namespace LeftBound12432
def owner : Owner := ⟨.program ⟨214⟩, ⟨15881⟩⟩
def transferEvent : Nat := 12432
def frameStart : Nat := 12336
def rule : BoundRule := .sum [.predecessor 0 12430 .coefficient, .predecessor 1 12431 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12430 .coefficient)
      LeftAuthority12428.bound (LeftAuthority12428.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12429RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12428.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12428.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12431 .coefficient)
      LeftBound12424.bound (LeftBound12424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12426RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority12428.bound, LeftBound12424.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority12428.bound, LeftBound12424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority12428.actual selector witness, LeftBound12424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12432

namespace LeftBound12436
def owner : Owner := ⟨.program ⟨214⟩, ⟨27706⟩⟩
def transferEvent : Nat := 12436
def frameStart : Nat := 12336
def rule : BoundRule := .sum [.predecessor 0 12434 .coefficient, .predecessor 1 12435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12434 .coefficient)
      LeftBound12432.bound (LeftBound12432.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12433RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12432.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12432.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12435 .coefficient)
      LeftBound12413.bound (LeftBound12413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12418RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12413.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12432.bound, LeftBound12413.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12432.bound, LeftBound12413.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12432.actual selector witness, LeftBound12413.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12436

namespace LeftBound12449
def owner : Owner := ⟨.program ⟨214⟩, ⟨27704⟩⟩
def transferEvent : Nat := 12449
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12447 .coefficient, .predecessor 1 12448 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 12447 .coefficient)
      LeftBound12278.bound (LeftBound12278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events048.exact12446RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 12448 .coefficient)
      LeftBound12261.bound (LeftBound12261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12268RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12261.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12261.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12278.bound, LeftBound12261.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12278.bound, LeftBound12261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound12278.actual selector witness, LeftBound12261.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12449

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
