import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard043
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard108

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound17718
def owner : Owner := ⟨.program ⟨214⟩, ⟨29651⟩⟩
def transferEvent : Nat := 17718
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17716 .coefficient) (.predecessor 1 17717 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17716 .coefficient)
      LeftBound17711.bound (LeftBound17711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17715RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17711.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17717 .coefficient)
      LeftBound5558.bound (LeftBound5558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17711.bound LeftBound5558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17711.bound, LeftBound5558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17711.actual selector witness) * (LeftBound5558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17718

namespace LeftBound17719
def owner : Owner := ⟨.program ⟨214⟩, ⟨29651⟩⟩
def transferEvent : Nat := 17719
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6661⟩⟩]⟩ [⟨.result 5555 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5555 .coefficient)
      LeftAuthority5554.bound (LeftAuthority5554.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6661⟩⟩) (rawTerms := some (Proof.Events021.exact5555RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5554.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5554.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5554.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5554.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5554.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17719

namespace LeftBound17720
def owner : Owner := ⟨.program ⟨214⟩, ⟨29651⟩⟩
def transferEvent : Nat := 17720
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17715 .summary) (.transfer 17719) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17715 .summary)
      LeftBound17714.bound (LeftBound17714.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29650⟩⟩) (rawTerms := some (Proof.Events069.exact17715RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17714.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17719)
      LeftBound17719.bound (LeftBound17719.actual selector witness) := by
  exact .transfer (LeftBound17719.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17714.bound LeftBound17719.bound
def bound : CoeffClass := .finite ⟨4743310290994884271912517632, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17714.bound, LeftBound17719.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17714.actual selector witness) * (LeftBound17719.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17720

namespace LeftBound17735
def owner : Owner := ⟨.program ⟨214⟩, ⟨29432⟩⟩
def transferEvent : Nat := 17735
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17733 .coefficient) (.predecessor 1 17734 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17733 .coefficient)
      LeftBound8246.bound (LeftBound8246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17734 .coefficient)
      LeftAuthority17731.bound (LeftAuthority17731.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17732RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17731.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8246.bound LeftAuthority17731.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8246.bound, LeftAuthority17731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8246.actual selector witness) * (LeftAuthority17731.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17735

namespace LeftBound17736
def owner : Owner := ⟨.program ⟨214⟩, ⟨29432⟩⟩
def transferEvent : Nat := 17736
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29430⟩⟩]⟩ [⟨.result 17732 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17732 .coefficient)
      LeftAuthority17731.bound (LeftAuthority17731.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29430⟩⟩) (rawTerms := some (Proof.Events069.exact17732RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17731.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17731.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17731.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17731.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17731.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17736

namespace LeftBound17737
def owner : Owner := ⟨.program ⟨214⟩, ⟨29432⟩⟩
def transferEvent : Nat := 17737
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8250 .summary) (.transfer 17736) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8250 .summary)
      LeftBound8249.bound (LeftBound8249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25549⟩⟩) (rawTerms := some (Proof.Events032.exact8250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17736)
      LeftBound17736.bound (LeftBound17736.actual selector witness) := by
  exact .transfer (LeftBound17736.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8249.bound LeftBound17736.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8249.bound, LeftBound17736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8249.actual selector witness) * (LeftBound17736.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17737

namespace LeftBound17748
def owner : Owner := ⟨.program ⟨214⟩, ⟨22354⟩⟩
def transferEvent : Nat := 17748
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 17746 .coefficient) (.value (.predecessor 1 17747 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17746 .coefficient)
      LeftAuthority17744.bound (LeftAuthority17744.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17744.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17747 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority17744.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17744.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17744.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound17748

namespace LeftBound17752
def owner : Owner := ⟨.program ⟨214⟩, ⟨22355⟩⟩
def transferEvent : Nat := 17752
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 17750 .coefficient) (.predecessor 1 17751 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17750 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17751 .coefficient)
      LeftBound17748.bound (LeftBound17748.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17748.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17748.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound17748.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound17748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound17748.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17752

namespace LeftBound17753
def owner : Owner := ⟨.program ⟨214⟩, ⟨22355⟩⟩
def transferEvent : Nat := 17753
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22352⟩⟩]⟩ [⟨.result 17745 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 17745 .coefficient)
      LeftAuthority17744.bound (LeftAuthority17744.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22352⟩⟩) (rawTerms := some (Proof.Events069.exact17745RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17744.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17744.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17744.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17744.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority17744.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound17753

namespace LeftBound17754
def owner : Owner := ⟨.program ⟨214⟩, ⟨22355⟩⟩
def transferEvent : Nat := 17754
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 17753) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 17753)
      LeftBound17753.bound (LeftBound17753.actual selector witness) := by
  exact .transfer (LeftBound17753.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound17753.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound17753.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound17753.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17754

namespace LeftBound17849
def owner : Owner := ⟨.program ⟨214⟩, ⟨16650⟩⟩
def transferEvent : Nat := 17849
def frameStart : Nat := 17810
def rule : BoundRule := .identity (.predecessor 0 17848 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17848 .coefficient)
      LeftAuthority17846.bound (LeftAuthority17846.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17847RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17846.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17846.derived selector witness)

def rawBound : CoeffClass := LeftAuthority17846.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17846.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority17846.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17849

namespace LeftBound17866
def owner : Owner := ⟨.program ⟨214⟩, ⟨16724⟩⟩
def transferEvent : Nat := 17866
def frameStart : Nat := 17810
def rule : BoundRule := .sum [.predecessor 0 17864 .coefficient, .predecessor 1 17865 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17864 .coefficient)
      LeftBound17849.bound (LeftBound17849.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17849.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17865 .coefficient)
      LeftAuthority17862.bound (LeftAuthority17862.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority17862.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound17849.bound, LeftAuthority17862.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17849.bound, LeftAuthority17862.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound17849.actual selector witness, LeftAuthority17862.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17866

namespace LeftBound17869
def owner : Owner := ⟨.program ⟨214⟩, ⟨16725⟩⟩
def transferEvent : Nat := 17869
def frameStart : Nat := 17810
def rule : BoundRule := .identity (.predecessor 0 17868 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17868 .coefficient)
      LeftBound17866.bound (LeftBound17866.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound17866.derived selector witness)

def rawBound : CoeffClass := LeftBound17866.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound17866.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound17869

namespace LeftBound17875
def owner : Owner := ⟨.program ⟨214⟩, ⟨16726⟩⟩
def transferEvent : Nat := 17875
def frameStart : Nat := 17810
def rule : BoundRule := .product (.predecessor 0 17873 .coefficient) (.predecessor 1 17874 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17873 .coefficient)
      LeftAuthority17871.bound (LeftAuthority17871.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17872RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17871.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17871.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17874 .coefficient)
      LeftBound17869.bound (LeftBound17869.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17869.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17869.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority17871.bound LeftBound17869.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17871.bound, LeftBound17869.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority17871.actual selector witness) * (LeftBound17869.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17875

namespace LeftBound17883
def owner : Owner := ⟨.program ⟨214⟩, ⟨16727⟩⟩
def transferEvent : Nat := 17883
def frameStart : Nat := 17810
def rule : BoundRule := .sum [.predecessor 0 17881 .coefficient, .predecessor 1 17882 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17881 .coefficient)
      LeftAuthority17879.bound (LeftAuthority17879.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17880RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17879.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17879.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17882 .coefficient)
      LeftBound17875.bound (LeftBound17875.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17877RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17875.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17875.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority17879.bound, LeftBound17875.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17879.bound, LeftBound17875.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority17879.actual selector witness, LeftBound17875.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound17883

namespace LeftBound17887
def owner : Owner := ⟨.program ⟨214⟩, ⟨29431⟩⟩
def transferEvent : Nat := 17887
def frameStart : Nat := 17810
def rule : BoundRule := .product (.predecessor 0 17885 .coefficient) (.predecessor 1 17886 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 17885 .coefficient)
      LeftBound17883.bound (LeftBound17883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17883.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17883.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 17886 .coefficient)
      LeftAuthority17860.bound (LeftAuthority17860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events069.exact17861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17860.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound17883.bound LeftAuthority17860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17883.bound, LeftAuthority17860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound17883.actual selector witness) * (LeftAuthority17860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound17887

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
