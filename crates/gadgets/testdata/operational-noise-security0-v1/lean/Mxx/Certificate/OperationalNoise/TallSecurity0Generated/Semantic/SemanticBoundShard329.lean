import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard021
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard297
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard328

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound49727
def owner : Owner := ⟨.program ⟨214⟩, ⟨26804⟩⟩
def transferEvent : Nat := 49727
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49725 .coefficient) (.predecessor 1 49726 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49725 .coefficient)
      LeftBound49720.bound (LeftBound49720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49720.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49726 .coefficient)
      LeftBound5818.bound (LeftBound5818.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5819RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5818.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5818.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49720.bound LeftBound5818.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49720.bound, LeftBound5818.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49720.actual selector witness) * (LeftBound5818.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49727

namespace LeftBound49728
def owner : Owner := ⟨.program ⟨214⟩, ⟨26804⟩⟩
def transferEvent : Nat := 49728
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6663⟩⟩]⟩ [⟨.result 5815 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5815 .coefficient)
      LeftAuthority5814.bound (LeftAuthority5814.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6663⟩⟩) (rawTerms := some (Proof.Events022.exact5815RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5814.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5814.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5814.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5814.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5814.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49728

namespace LeftBound49729
def owner : Owner := ⟨.program ⟨214⟩, ⟨26804⟩⟩
def transferEvent : Nat := 49729
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 49724 .summary) (.transfer 49728) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49724 .summary)
      LeftBound49723.bound (LeftBound49723.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26803⟩⟩) (rawTerms := some (Proof.Events194.exact49724RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound49723.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49728)
      LeftBound49728.bound (LeftBound49728.actual selector witness) := by
  exact .transfer (LeftBound49728.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49723.bound LeftBound49728.bound
def bound : CoeffClass := .finite ⟨4741336194231092170536779776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49723.bound, LeftBound49728.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49723.actual selector witness) * (LeftBound49728.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49729

namespace LeftBound49744
def owner : Owner := ⟨.program ⟨214⟩, ⟨26585⟩⟩
def transferEvent : Nat := 49744
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49742 .coefficient) (.predecessor 1 49743 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49742 .coefficient)
      LeftBound44031.bound (LeftBound44031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44035RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49743 .coefficient)
      LeftAuthority49740.bound (LeftAuthority49740.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49741RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49740.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44031.bound LeftAuthority49740.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44031.bound, LeftAuthority49740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44031.actual selector witness) * (LeftAuthority49740.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49744

namespace LeftBound49745
def owner : Owner := ⟨.program ⟨214⟩, ⟨26585⟩⟩
def transferEvent : Nat := 49745
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26583⟩⟩]⟩ [⟨.result 49741 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49741 .coefficient)
      LeftAuthority49740.bound (LeftAuthority49740.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26583⟩⟩) (rawTerms := some (Proof.Events194.exact49741RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49740.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49740.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49740.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49740.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49740.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49745

namespace LeftBound49746
def owner : Owner := ⟨.program ⟨214⟩, ⟨26585⟩⟩
def transferEvent : Nat := 49746
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44035 .summary) (.transfer 49745) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44035 .summary)
      LeftBound44034.bound (LeftBound44034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25000⟩⟩) (rawTerms := some (Proof.Events172.exact44035RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49745)
      LeftBound49745.bound (LeftBound49745.actual selector witness) := by
  exact .transfer (LeftBound49745.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44034.bound LeftBound49745.bound
def bound : CoeffClass := .finite ⟨1291900378790628425728, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44034.bound, LeftBound49745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44034.actual selector witness) * (LeftBound49745.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49746

namespace LeftBound49757
def owner : Owner := ⟨.program ⟨214⟩, ⟨20474⟩⟩
def transferEvent : Nat := 49757
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 49755 .coefficient) (.value (.predecessor 1 49756 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49755 .coefficient)
      LeftAuthority49753.bound (LeftAuthority49753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49754RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49756 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority49753.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49753.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49753.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound49757

namespace LeftBound49761
def owner : Owner := ⟨.program ⟨214⟩, ⟨20475⟩⟩
def transferEvent : Nat := 49761
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 49759 .coefficient) (.predecessor 1 49760 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49759 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49760 .coefficient)
      LeftBound49757.bound (LeftBound49757.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49758RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49757.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49757.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound49757.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound49757.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound49757.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49761

namespace LeftBound49762
def owner : Owner := ⟨.program ⟨214⟩, ⟨20475⟩⟩
def transferEvent : Nat := 49762
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20472⟩⟩]⟩ [⟨.result 49754 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 49754 .coefficient)
      LeftAuthority49753.bound (LeftAuthority49753.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20472⟩⟩) (rawTerms := some (Proof.Events194.exact49754RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49753.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49753.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority49753.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority49753.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound49762

namespace LeftBound49763
def owner : Owner := ⟨.program ⟨214⟩, ⟨20475⟩⟩
def transferEvent : Nat := 49763
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 49762) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 49762)
      LeftBound49762.bound (LeftBound49762.actual selector witness) := by
  exact .transfer (LeftBound49762.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound49762.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound49762.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound49762.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49763

namespace LeftBound49858
def owner : Owner := ⟨.program ⟨214⟩, ⟨14962⟩⟩
def transferEvent : Nat := 49858
def frameStart : Nat := 49819
def rule : BoundRule := .identity (.predecessor 0 49857 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49857 .coefficient)
      LeftAuthority49855.bound (LeftAuthority49855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49855.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49855.derived selector witness)

def rawBound : CoeffClass := LeftAuthority49855.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority49855.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49858

namespace LeftBound49875
def owner : Owner := ⟨.program ⟨214⟩, ⟨15001⟩⟩
def transferEvent : Nat := 49875
def frameStart : Nat := 49819
def rule : BoundRule := .sum [.predecessor 0 49873 .coefficient, .predecessor 1 49874 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49873 .coefficient)
      LeftBound49858.bound (LeftBound49858.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49874 .coefficient)
      LeftAuthority49871.bound (LeftAuthority49871.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority49871.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound49858.bound, LeftAuthority49871.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49858.bound, LeftAuthority49871.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound49858.actual selector witness, LeftAuthority49871.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49875

namespace LeftBound49878
def owner : Owner := ⟨.program ⟨214⟩, ⟨15002⟩⟩
def transferEvent : Nat := 49878
def frameStart : Nat := 49819
def rule : BoundRule := .identity (.predecessor 0 49877 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49877 .coefficient)
      LeftBound49875.bound (LeftBound49875.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound49875.derived selector witness)

def rawBound : CoeffClass := LeftBound49875.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound49875.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound49878

namespace LeftBound49884
def owner : Owner := ⟨.program ⟨214⟩, ⟨15003⟩⟩
def transferEvent : Nat := 49884
def frameStart : Nat := 49819
def rule : BoundRule := .product (.predecessor 0 49882 .coefficient) (.predecessor 1 49883 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49882 .coefficient)
      LeftAuthority49880.bound (LeftAuthority49880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49880.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49883 .coefficient)
      LeftBound49878.bound (LeftBound49878.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49879RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49878.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49878.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority49880.bound LeftBound49878.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49880.bound, LeftBound49878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority49880.actual selector witness) * (LeftBound49878.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49884

namespace LeftBound49892
def owner : Owner := ⟨.program ⟨214⟩, ⟨15004⟩⟩
def transferEvent : Nat := 49892
def frameStart : Nat := 49819
def rule : BoundRule := .sum [.predecessor 0 49890 .coefficient, .predecessor 1 49891 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49890 .coefficient)
      LeftAuthority49888.bound (LeftAuthority49888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49889RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49888.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49891 .coefficient)
      LeftBound49884.bound (LeftBound49884.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49886RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49884.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49884.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority49888.bound, LeftBound49884.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority49888.bound, LeftBound49884.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority49888.actual selector witness, LeftBound49884.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound49892

namespace LeftBound49896
def owner : Owner := ⟨.program ⟨214⟩, ⟨26584⟩⟩
def transferEvent : Nat := 49896
def frameStart : Nat := 49819
def rule : BoundRule := .product (.predecessor 0 49894 .coefficient) (.predecessor 1 49895 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 49894 .coefficient)
      LeftBound49892.bound (LeftBound49892.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49893RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound49892.bound, RecordedBoundRefines] <;> decide)
      (LeftBound49892.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 49895 .coefficient)
      LeftAuthority49869.bound (LeftAuthority49869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events194.exact49870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority49869.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority49869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound49892.bound LeftAuthority49869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound49892.bound, LeftAuthority49869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound49892.actual selector witness) * (LeftAuthority49869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound49896

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
