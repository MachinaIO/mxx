import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard320

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48448
def owner : Owner := ⟨.program ⟨214⟩, ⟨28105⟩⟩
def transferEvent : Nat := 48448
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 48446 .coefficient, .predecessor 1 48447 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48446 .coefficient)
      LeftBound48277.bound (LeftBound48277.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48445RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48277.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48277.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48447 .coefficient)
      LeftBound48260.bound (LeftBound48260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48260.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48260.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48277.bound, LeftBound48260.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48277.bound, LeftBound48260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48277.actual selector witness, LeftBound48260.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48448

namespace LeftBound48451
def owner : Owner := ⟨.program ⟨214⟩, ⟨28105⟩⟩
def transferEvent : Nat := 48451
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 48445 .summary, .result 48267 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48445 .summary)
      LeftBound48279.bound (LeftBound48279.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21483⟩⟩) (rawTerms := some (Proof.Events189.exact48445RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48267 .summary)
      LeftBound48262.bound (LeftBound48262.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28104⟩⟩) (rawTerms := some (Proof.Events188.exact48267RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48262.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48279.bound, LeftBound48262.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48279.bound, LeftBound48262.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48279.actual selector witness, LeftBound48262.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48451

namespace LeftBound48455
def owner : Owner := ⟨.program ⟨214⟩, ⟨28106⟩⟩
def transferEvent : Nat := 48455
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48453 .coefficient) (.predecessor 1 48454 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48453 .coefficient)
      LeftBound48448.bound (LeftBound48448.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48448.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48448.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48454 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48448.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48448.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48448.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48455

namespace LeftBound48456
def owner : Owner := ⟨.program ⟨214⟩, ⟨28106⟩⟩
def transferEvent : Nat := 48456
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6637⟩⟩]⟩ [⟨.result 5695 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5695 .coefficient)
      LeftAuthority5694.bound (LeftAuthority5694.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6637⟩⟩) (rawTerms := some (Proof.Events022.exact5695RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5694.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5694.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5694.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5694.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5694.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48456

namespace LeftBound48457
def owner : Owner := ⟨.program ⟨214⟩, ⟨28106⟩⟩
def transferEvent : Nat := 48457
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 48452 .summary) (.transfer 48456) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48452 .summary)
      LeftBound48451.bound (LeftBound48451.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28105⟩⟩) (rawTerms := some (Proof.Events189.exact48452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound48451.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48456)
      LeftBound48456.bound (LeftBound48456.actual selector witness) := by
  exact .transfer (LeftBound48456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48451.bound LeftBound48456.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48451.bound, LeftBound48456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48451.actual selector witness) * (LeftBound48456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48457

namespace LeftBound48472
def owner : Owner := ⟨.program ⟨214⟩, ⟨27887⟩⟩
def transferEvent : Nat := 48472
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48470 .coefficient) (.predecessor 1 48471 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48470 .coefficient)
      LeftBound41139.bound (LeftBound41139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events160.exact41143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound41139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound41139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48471 .coefficient)
      LeftAuthority48468.bound (LeftAuthority48468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48468.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41139.bound LeftAuthority48468.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41139.bound, LeftAuthority48468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41139.actual selector witness) * (LeftAuthority48468.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48472

namespace LeftBound48473
def owner : Owner := ⟨.program ⟨214⟩, ⟨27887⟩⟩
def transferEvent : Nat := 48473
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27885⟩⟩]⟩ [⟨.result 48469 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48469 .coefficient)
      LeftAuthority48468.bound (LeftAuthority48468.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27885⟩⟩) (rawTerms := some (Proof.Events189.exact48469RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48468.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48468.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48468.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48468.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48473

namespace LeftBound48474
def owner : Owner := ⟨.program ⟨214⟩, ⟨27887⟩⟩
def transferEvent : Nat := 48474
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 41143 .summary) (.transfer 48473) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 41143 .summary)
      LeftBound41142.bound (LeftBound41142.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26078⟩⟩) (rawTerms := some (Proof.Events160.exact41143RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound41142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48473)
      LeftBound48473.bound (LeftBound48473.actual selector witness) := by
  exact .transfer (LeftBound48473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound41142.bound LeftBound48473.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound41142.bound, LeftBound48473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound41142.actual selector witness) * (LeftBound48473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48474

namespace LeftBound48485
def owner : Owner := ⟨.program ⟨214⟩, ⟨21338⟩⟩
def transferEvent : Nat := 48485
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48483 .coefficient) (.value (.predecessor 1 48484 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48483 .coefficient)
      LeftAuthority48481.bound (LeftAuthority48481.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48482RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48481.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48484 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48481.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48481.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48481.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48485

namespace LeftBound48489
def owner : Owner := ⟨.program ⟨214⟩, ⟨21339⟩⟩
def transferEvent : Nat := 48489
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48487 .coefficient) (.predecessor 1 48488 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48487 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48488 .coefficient)
      LeftBound48485.bound (LeftBound48485.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48486RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48485.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48485.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound48485.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound48485.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound48485.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48489

namespace LeftBound48490
def owner : Owner := ⟨.program ⟨214⟩, ⟨21339⟩⟩
def transferEvent : Nat := 48490
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21336⟩⟩]⟩ [⟨.result 48482 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48482 .coefficient)
      LeftAuthority48481.bound (LeftAuthority48481.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21336⟩⟩) (rawTerms := some (Proof.Events189.exact48482RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48481.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48481.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48481.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48481.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48481.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48490

namespace LeftBound48491
def owner : Owner := ⟨.program ⟨214⟩, ⟨21339⟩⟩
def transferEvent : Nat := 48491
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 48490) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48490)
      LeftBound48490.bound (LeftBound48490.actual selector witness) := by
  exact .transfer (LeftBound48490.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound48490.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound48490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound48490.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48491

namespace LeftBound48586
def owner : Owner := ⟨.program ⟨214⟩, ⟨15949⟩⟩
def transferEvent : Nat := 48586
def frameStart : Nat := 48547
def rule : BoundRule := .identity (.predecessor 0 48585 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48585 .coefficient)
      LeftAuthority48583.bound (LeftAuthority48583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48583.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48583.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48583.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48583.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority48583.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48586

namespace LeftBound48603
def owner : Owner := ⟨.program ⟨214⟩, ⟨16023⟩⟩
def transferEvent : Nat := 48603
def frameStart : Nat := 48547
def rule : BoundRule := .sum [.predecessor 0 48601 .coefficient, .predecessor 1 48602 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48601 .coefficient)
      LeftBound48586.bound (LeftBound48586.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48602 .coefficient)
      LeftAuthority48599.bound (LeftAuthority48599.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48599.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48586.bound, LeftAuthority48599.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48586.bound, LeftAuthority48599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48586.actual selector witness, LeftAuthority48599.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48603

namespace LeftBound48606
def owner : Owner := ⟨.program ⟨214⟩, ⟨16024⟩⟩
def transferEvent : Nat := 48606
def frameStart : Nat := 48547
def rule : BoundRule := .identity (.predecessor 0 48605 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48605 .coefficient)
      LeftBound48603.bound (LeftBound48603.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48603.derived selector witness)

def rawBound : CoeffClass := LeftBound48603.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound48603.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48606

namespace LeftBound48612
def owner : Owner := ⟨.program ⟨214⟩, ⟨16025⟩⟩
def transferEvent : Nat := 48612
def frameStart : Nat := 48547
def rule : BoundRule := .product (.predecessor 0 48610 .coefficient) (.predecessor 1 48611 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48610 .coefficient)
      LeftAuthority48608.bound (LeftAuthority48608.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48609RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48608.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48608.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48611 .coefficient)
      LeftBound48606.bound (LeftBound48606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48607RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48606.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority48608.bound LeftBound48606.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48608.bound, LeftBound48606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority48608.actual selector witness) * (LeftBound48606.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48612

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
