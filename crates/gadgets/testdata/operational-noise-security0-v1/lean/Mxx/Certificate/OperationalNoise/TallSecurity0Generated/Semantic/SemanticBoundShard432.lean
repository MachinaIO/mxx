import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard402
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard431

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound64582
def owner : Owner := ⟨.program ⟨214⟩, ⟨26365⟩⟩
def transferEvent : Nat := 64582
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26363⟩⟩]⟩ [⟨.result 64578 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64578 .coefficient)
      LeftAuthority64577.bound (LeftAuthority64577.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26363⟩⟩) (rawTerms := some (Proof.Events252.exact64578RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64577.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64577.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64577.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64577.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64577.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64582

namespace LeftBound64583
def owner : Owner := ⟨.program ⟨214⟩, ⟨26365⟩⟩
def transferEvent : Nat := 64583
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 59142 .summary) (.transfer 64582) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 59142 .summary)
      LeftBound59141.bound (LeftBound59141.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24918⟩⟩) (rawTerms := some (Proof.Events231.exact59142RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound59141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64582)
      LeftBound64582.bound (LeftBound64582.actual selector witness) := by
  exact .transfer (LeftBound64582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound59141.bound LeftBound64582.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound59141.bound, LeftBound64582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound59141.actual selector witness) * (LeftBound64582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64583

namespace LeftBound64594
def owner : Owner := ⟨.program ⟨214⟩, ⟨20326⟩⟩
def transferEvent : Nat := 64594
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 64592 .coefficient) (.value (.predecessor 1 64593 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64592 .coefficient)
      LeftAuthority64590.bound (LeftAuthority64590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64593 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority64590.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64590.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64590.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound64594

namespace LeftBound64598
def owner : Owner := ⟨.program ⟨214⟩, ⟨20327⟩⟩
def transferEvent : Nat := 64598
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 64596 .coefficient) (.predecessor 1 64597 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64596 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64597 .coefficient)
      LeftBound64594.bound (LeftBound64594.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64595RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64594.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64594.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound64594.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound64594.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound64594.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64598

namespace LeftBound64599
def owner : Owner := ⟨.program ⟨214⟩, ⟨20327⟩⟩
def transferEvent : Nat := 64599
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20324⟩⟩]⟩ [⟨.result 64591 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 64591 .coefficient)
      LeftAuthority64590.bound (LeftAuthority64590.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20324⟩⟩) (rawTerms := some (Proof.Events252.exact64591RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64590.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64590.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority64590.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority64590.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound64599

namespace LeftBound64600
def owner : Owner := ⟨.program ⟨214⟩, ⟨20327⟩⟩
def transferEvent : Nat := 64600
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 64599) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 64599)
      LeftBound64599.bound (LeftBound64599.actual selector witness) := by
  exact .transfer (LeftBound64599.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound64599.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound64599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound64599.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64600

namespace LeftBound64695
def owner : Owner := ⟨.program ⟨214⟩, ⟨14797⟩⟩
def transferEvent : Nat := 64695
def frameStart : Nat := 64656
def rule : BoundRule := .identity (.predecessor 0 64694 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64694 .coefficient)
      LeftAuthority64692.bound (LeftAuthority64692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64692.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64692.derived selector witness)

def rawBound : CoeffClass := LeftAuthority64692.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64692.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority64692.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64695

namespace LeftBound64712
def owner : Owner := ⟨.program ⟨214⟩, ⟨14836⟩⟩
def transferEvent : Nat := 64712
def frameStart : Nat := 64656
def rule : BoundRule := .sum [.predecessor 0 64710 .coefficient, .predecessor 1 64711 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64710 .coefficient)
      LeftBound64695.bound (LeftBound64695.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64695.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64711 .coefficient)
      LeftAuthority64708.bound (LeftAuthority64708.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority64708.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64695.bound, LeftAuthority64708.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64695.bound, LeftAuthority64708.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64695.actual selector witness, LeftAuthority64708.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64712

namespace LeftBound64715
def owner : Owner := ⟨.program ⟨214⟩, ⟨14837⟩⟩
def transferEvent : Nat := 64715
def frameStart : Nat := 64656
def rule : BoundRule := .identity (.predecessor 0 64714 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64714 .coefficient)
      LeftBound64712.bound (LeftBound64712.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound64712.derived selector witness)

def rawBound : CoeffClass := LeftBound64712.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound64712.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound64715

namespace LeftBound64721
def owner : Owner := ⟨.program ⟨214⟩, ⟨14838⟩⟩
def transferEvent : Nat := 64721
def frameStart : Nat := 64656
def rule : BoundRule := .product (.predecessor 0 64719 .coefficient) (.predecessor 1 64720 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64719 .coefficient)
      LeftAuthority64717.bound (LeftAuthority64717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64720 .coefficient)
      LeftBound64715.bound (LeftBound64715.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64715.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64715.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority64717.bound LeftBound64715.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64717.bound, LeftBound64715.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority64717.actual selector witness) * (LeftBound64715.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64721

namespace LeftBound64729
def owner : Owner := ⟨.program ⟨214⟩, ⟨14839⟩⟩
def transferEvent : Nat := 64729
def frameStart : Nat := 64656
def rule : BoundRule := .sum [.predecessor 0 64727 .coefficient, .predecessor 1 64728 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64727 .coefficient)
      LeftAuthority64725.bound (LeftAuthority64725.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64726RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64725.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64725.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64728 .coefficient)
      LeftBound64721.bound (LeftBound64721.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64723RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64721.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64721.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64725.bound, LeftBound64721.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64725.bound, LeftBound64721.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64725.actual selector witness, LeftBound64721.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64729

namespace LeftBound64733
def owner : Owner := ⟨.program ⟨214⟩, ⟨26364⟩⟩
def transferEvent : Nat := 64733
def frameStart : Nat := 64656
def rule : BoundRule := .product (.predecessor 0 64731 .coefficient) (.predecessor 1 64732 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64731 .coefficient)
      LeftBound64729.bound (LeftBound64729.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64730RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64729.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64729.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64732 .coefficient)
      LeftAuthority64706.bound (LeftAuthority64706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64706.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound64729.bound LeftAuthority64706.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64729.bound, LeftAuthority64706.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound64729.actual selector witness) * (LeftAuthority64706.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64733

namespace LeftBound64744
def owner : Owner := ⟨.program ⟨214⟩, ⟨14894⟩⟩
def transferEvent : Nat := 64744
def frameStart : Nat := 64656
def rule : BoundRule := .product (.predecessor 0 64742 .coefficient) (.predecessor 1 64743 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64742 .coefficient)
      LeftAuthority64717.bound (LeftAuthority64717.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64718RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64717.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64717.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64743 .coefficient)
      LeftAuthority64740.bound (LeftAuthority64740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority64717.bound LeftAuthority64740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64717.bound, LeftAuthority64740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority64717.actual selector witness) * (LeftAuthority64740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound64744

namespace LeftBound64752
def owner : Owner := ⟨.program ⟨214⟩, ⟨14895⟩⟩
def transferEvent : Nat := 64752
def frameStart : Nat := 64656
def rule : BoundRule := .sum [.predecessor 0 64750 .coefficient, .predecessor 1 64751 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64750 .coefficient)
      LeftAuthority64748.bound (LeftAuthority64748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority64748.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority64748.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64751 .coefficient)
      LeftBound64744.bound (LeftBound64744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64746RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64744.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64744.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority64748.bound, LeftBound64744.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority64748.bound, LeftBound64744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority64748.actual selector witness, LeftBound64744.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64752

namespace LeftBound64756
def owner : Owner := ⟨.program ⟨214⟩, ⟨26369⟩⟩
def transferEvent : Nat := 64756
def frameStart : Nat := 64656
def rule : BoundRule := .sum [.predecessor 0 64754 .coefficient, .predecessor 1 64755 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64754 .coefficient)
      LeftBound64752.bound (LeftBound64752.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64753RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64752.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64752.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64755 .coefficient)
      LeftBound64733.bound (LeftBound64733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64738RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64752.bound, LeftBound64733.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64752.bound, LeftBound64733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64752.actual selector witness, LeftBound64733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64756

namespace LeftBound64769
def owner : Owner := ⟨.program ⟨214⟩, ⟨26366⟩⟩
def transferEvent : Nat := 64769
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 64767 .coefficient, .predecessor 1 64768 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 64767 .coefficient)
      LeftBound64598.bound (LeftBound64598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64766RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64598.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 64768 .coefficient)
      LeftBound64581.bound (LeftBound64581.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events252.exact64588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound64581.bound, RecordedBoundRefines] <;> decide)
      (LeftBound64581.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound64598.bound, LeftBound64581.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound64598.bound, LeftBound64581.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound64598.actual selector witness, LeftBound64581.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound64769

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
