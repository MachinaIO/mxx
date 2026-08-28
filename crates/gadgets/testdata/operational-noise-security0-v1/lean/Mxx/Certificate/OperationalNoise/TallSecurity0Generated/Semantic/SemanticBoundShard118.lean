import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard019
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard070
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard117

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound19182
def owner : Owner := ⟨.program ⟨214⟩, ⟨28134⟩⟩
def transferEvent : Nat := 19182
def frameStart : Nat := 19082
def rule : BoundRule := .sum [.predecessor 0 19180 .coefficient, .predecessor 1 19181 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19180 .coefficient)
      LeftBound19178.bound (LeftBound19178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19181 .coefficient)
      LeftBound19159.bound (LeftBound19159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19159.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19178.bound, LeftBound19159.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19178.bound, LeftBound19159.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19178.actual selector witness, LeftBound19159.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19182

namespace LeftBound19195
def owner : Owner := ⟨.program ⟨214⟩, ⟨28131⟩⟩
def transferEvent : Nat := 19195
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 19193 .coefficient, .predecessor 1 19194 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19193 .coefficient)
      LeftBound19024.bound (LeftBound19024.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19024.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19024.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19194 .coefficient)
      LeftBound19007.bound (LeftBound19007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19014RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19007.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19024.bound, LeftBound19007.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19024.bound, LeftBound19007.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19024.actual selector witness, LeftBound19007.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19195

namespace LeftBound19198
def owner : Owner := ⟨.program ⟨214⟩, ⟨28131⟩⟩
def transferEvent : Nat := 19198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 19192 .summary, .result 19014 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19192 .summary)
      LeftBound19026.bound (LeftBound19026.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨21491⟩⟩) (rawTerms := some (Proof.Events074.exact19192RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19026.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19014 .summary)
      LeftBound19009.bound (LeftBound19009.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28130⟩⟩) (rawTerms := some (Proof.Events074.exact19014RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19009.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19026.bound, LeftBound19009.bound]
def bound : CoeffClass := .finite ⟨1292113298829627502592, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19026.bound, LeftBound19009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19026.actual selector witness, LeftBound19009.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19198

namespace LeftBound19202
def owner : Owner := ⟨.program ⟨214⟩, ⟨28132⟩⟩
def transferEvent : Nat := 19202
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19200 .coefficient) (.predecessor 1 19201 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19200 .coefficient)
      LeftBound19195.bound (LeftBound19195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19199RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19195.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19201 .coefficient)
      LeftBound5698.bound (LeftBound5698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events022.exact5699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19195.bound LeftBound5698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19195.bound, LeftBound5698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19195.actual selector witness) * (LeftBound5698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19202

namespace LeftBound19203
def owner : Owner := ⟨.program ⟨214⟩, ⟨28132⟩⟩
def transferEvent : Nat := 19203
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
end LeftBound19203

namespace LeftBound19204
def owner : Owner := ⟨.program ⟨214⟩, ⟨28132⟩⟩
def transferEvent : Nat := 19204
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 19199 .summary) (.transfer 19203) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19199 .summary)
      LeftBound19198.bound (LeftBound19198.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28131⟩⟩) (rawTerms := some (Proof.Events074.exact19199RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound19198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19203)
      LeftBound19203.bound (LeftBound19203.actual selector witness) := by
  exact .transfer (LeftBound19203.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19198.bound LeftBound19203.bound
def bound : CoeffClass := .finite ⟨4742076480517514208552681472, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19198.bound, LeftBound19203.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19198.actual selector witness) * (LeftBound19203.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19204

namespace LeftBound19219
def owner : Owner := ⟨.program ⟨214⟩, ⟨27913⟩⟩
def transferEvent : Nat := 19219
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19217 .coefficient) (.predecessor 1 19218 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19217 .coefficient)
      LeftBound11753.bound (LeftBound11753.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events045.exact11757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11753.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11753.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19218 .coefficient)
      LeftAuthority19215.bound (LeftAuthority19215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19215.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11753.bound LeftAuthority19215.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11753.bound, LeftAuthority19215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11753.actual selector witness) * (LeftAuthority19215.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19219

namespace LeftBound19220
def owner : Owner := ⟨.program ⟨214⟩, ⟨27913⟩⟩
def transferEvent : Nat := 19220
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨27911⟩⟩]⟩ [⟨.result 19216 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19216 .coefficient)
      LeftAuthority19215.bound (LeftAuthority19215.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨27911⟩⟩) (rawTerms := some (Proof.Events075.exact19216RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19215.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19215.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19215.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19215.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19220

namespace LeftBound19221
def owner : Owner := ⟨.program ⟨214⟩, ⟨27913⟩⟩
def transferEvent : Nat := 19221
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11757 .summary) (.transfer 19220) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11757 .summary)
      LeftBound11756.bound (LeftBound11756.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26088⟩⟩) (rawTerms := some (Proof.Events045.exact11757RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11756.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19220)
      LeftBound19220.bound (LeftBound19220.actual selector witness) := by
  exact .transfer (LeftBound19220.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11756.bound LeftBound19220.bound
def bound : CoeffClass := .finite ⟨1292068472128282820608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11756.bound, LeftBound19220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11756.actual selector witness) * (LeftBound19220.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19221

namespace LeftBound19232
def owner : Owner := ⟨.program ⟨214⟩, ⟨21346⟩⟩
def transferEvent : Nat := 19232
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 19230 .coefficient) (.value (.predecessor 1 19231 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19230 .coefficient)
      LeftAuthority19228.bound (LeftAuthority19228.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19229RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19228.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19228.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19231 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority19228.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19228.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19228.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound19232

namespace LeftBound19236
def owner : Owner := ⟨.program ⟨214⟩, ⟨21347⟩⟩
def transferEvent : Nat := 19236
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19234 .coefficient) (.predecessor 1 19235 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19234 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19235 .coefficient)
      LeftBound19232.bound (LeftBound19232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19232.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19232.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound19232.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound19232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound19232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19236

namespace LeftBound19237
def owner : Owner := ⟨.program ⟨214⟩, ⟨21347⟩⟩
def transferEvent : Nat := 19237
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21344⟩⟩]⟩ [⟨.result 19229 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19229 .coefficient)
      LeftAuthority19228.bound (LeftAuthority19228.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21344⟩⟩) (rawTerms := some (Proof.Events075.exact19229RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19228.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19228.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19228.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19228.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19237

namespace LeftBound19238
def owner : Owner := ⟨.program ⟨214⟩, ⟨21347⟩⟩
def transferEvent : Nat := 19238
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 19237) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19237)
      LeftBound19237.bound (LeftBound19237.actual selector witness) := by
  exact .transfer (LeftBound19237.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound19237.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound19237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound19237.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19238

namespace LeftBound19333
def owner : Owner := ⟨.program ⟨214⟩, ⟨15957⟩⟩
def transferEvent : Nat := 19333
def frameStart : Nat := 19294
def rule : BoundRule := .identity (.predecessor 0 19332 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19332 .coefficient)
      LeftAuthority19330.bound (LeftAuthority19330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events075.exact19331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19330.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19330.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19330.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19330.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19333

namespace LeftBound19350
def owner : Owner := ⟨.program ⟨214⟩, ⟨16031⟩⟩
def transferEvent : Nat := 19350
def frameStart : Nat := 19294
def rule : BoundRule := .sum [.predecessor 0 19348 .coefficient, .predecessor 1 19349 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19348 .coefficient)
      LeftBound19333.bound (LeftBound19333.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19349 .coefficient)
      LeftAuthority19346.bound (LeftAuthority19346.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority19346.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19333.bound, LeftAuthority19346.bound]
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19333.bound, LeftAuthority19346.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19333.actual selector witness, LeftAuthority19346.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19350

namespace LeftBound19353
def owner : Owner := ⟨.program ⟨214⟩, ⟨16032⟩⟩
def transferEvent : Nat := 19353
def frameStart : Nat := 19294
def rule : BoundRule := .identity (.predecessor 0 19352 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19352 .coefficient)
      LeftBound19350.bound (LeftBound19350.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19350.derived selector witness)

def rawBound : CoeffClass := LeftBound19350.bound
def bound : CoeffClass := .finite ⟨18, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19350.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound19350.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19353

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
