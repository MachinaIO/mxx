import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard066
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard067
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard116

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound18992
def owner : Owner := ⟨.program ⟨214⟩, ⟨28349⟩⟩
def transferEvent : Nat := 18992
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 18987 .summary) (.transfer 18991) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 18987 .summary)
      LeftBound18986.bound (LeftBound18986.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨28348⟩⟩) (rawTerms := some (Proof.Events074.exact18987RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound18986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 18991)
      LeftBound18991.bound (LeftBound18991.actual selector witness) := by
  exact .transfer (LeftBound18991.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound18986.bound LeftBound18991.bound
def bound : CoeffClass := .finite ⟨4742323242612988221224648704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound18986.bound, LeftBound18991.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound18986.actual selector witness) * (LeftBound18991.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound18992

namespace LeftBound19007
def owner : Owner := ⟨.program ⟨214⟩, ⟨28130⟩⟩
def transferEvent : Nat := 19007
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19005 .coefficient) (.predecessor 1 19006 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19005 .coefficient)
      LeftBound11252.bound (LeftBound11252.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events043.exact11256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound11252.bound, RecordedBoundRefines] <;> decide)
      (LeftBound11252.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19006 .coefficient)
      LeftAuthority19003.bound (LeftAuthority19003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19003.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11252.bound LeftAuthority19003.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11252.bound, LeftAuthority19003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11252.actual selector witness) * (LeftAuthority19003.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19007

namespace LeftBound19008
def owner : Owner := ⟨.program ⟨214⟩, ⟨28130⟩⟩
def transferEvent : Nat := 19008
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28128⟩⟩]⟩ [⟨.result 19004 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19004 .coefficient)
      LeftAuthority19003.bound (LeftAuthority19003.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28128⟩⟩) (rawTerms := some (Proof.Events074.exact19004RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19003.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19003.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19003.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19003.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19008

namespace LeftBound19009
def owner : Owner := ⟨.program ⟨214⟩, ⟨28130⟩⟩
def transferEvent : Nat := 19009
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 11256 .summary) (.transfer 19008) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 11256 .summary)
      LeftBound11255.bound (LeftBound11255.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26165⟩⟩) (rawTerms := some (Proof.Events043.exact11256RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound11255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19008)
      LeftBound19008.bound (LeftBound19008.actual selector witness) := by
  exact .transfer (LeftBound19008.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound11255.bound LeftBound19008.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound11255.bound, LeftBound19008.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound11255.actual selector witness) * (LeftBound19008.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19009

namespace LeftBound19020
def owner : Owner := ⟨.program ⟨214⟩, ⟨21490⟩⟩
def transferEvent : Nat := 19020
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 19018 .coefficient) (.value (.predecessor 1 19019 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19018 .coefficient)
      LeftAuthority19016.bound (LeftAuthority19016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19017RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19019 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority19016.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19016.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19016.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound19020

namespace LeftBound19024
def owner : Owner := ⟨.program ⟨214⟩, ⟨21491⟩⟩
def transferEvent : Nat := 19024
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 19022 .coefficient) (.predecessor 1 19023 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19022 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19023 .coefficient)
      LeftBound19020.bound (LeftBound19020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19021RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19020.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound19020.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound19020.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound19020.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19024

namespace LeftBound19025
def owner : Owner := ⟨.program ⟨214⟩, ⟨21491⟩⟩
def transferEvent : Nat := 19025
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21488⟩⟩]⟩ [⟨.result 19017 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 19017 .coefficient)
      LeftAuthority19016.bound (LeftAuthority19016.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21488⟩⟩) (rawTerms := some (Proof.Events074.exact19017RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19016.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19016.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority19016.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19016.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority19016.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound19025

namespace LeftBound19026
def owner : Owner := ⟨.program ⟨214⟩, ⟨21491⟩⟩
def transferEvent : Nat := 19026
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 19025) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 19025)
      LeftBound19025.bound (LeftBound19025.actual selector witness) := by
  exact .transfer (LeftBound19025.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound19025.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound19025.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound19025.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19026

namespace LeftBound19121
def owner : Owner := ⟨.program ⟨214⟩, ⟨16076⟩⟩
def transferEvent : Nat := 19121
def frameStart : Nat := 19082
def rule : BoundRule := .identity (.predecessor 0 19120 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19120 .coefficient)
      LeftAuthority19118.bound (LeftAuthority19118.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19119RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19118.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19118.derived selector witness)

def rawBound : CoeffClass := LeftAuthority19118.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19118.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority19118.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19121

namespace LeftBound19138
def owner : Owner := ⟨.program ⟨214⟩, ⟨16150⟩⟩
def transferEvent : Nat := 19138
def frameStart : Nat := 19082
def rule : BoundRule := .sum [.predecessor 0 19136 .coefficient, .predecessor 1 19137 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19136 .coefficient)
      LeftBound19121.bound (LeftBound19121.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19137 .coefficient)
      LeftAuthority19134.bound (LeftAuthority19134.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority19134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound19121.bound, LeftAuthority19134.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19121.bound, LeftAuthority19134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound19121.actual selector witness, LeftAuthority19134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19138

namespace LeftBound19141
def owner : Owner := ⟨.program ⟨214⟩, ⟨16151⟩⟩
def transferEvent : Nat := 19141
def frameStart : Nat := 19082
def rule : BoundRule := .identity (.predecessor 0 19140 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19140 .coefficient)
      LeftBound19138.bound (LeftBound19138.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound19138.derived selector witness)

def rawBound : CoeffClass := LeftBound19138.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound19138.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound19141

namespace LeftBound19147
def owner : Owner := ⟨.program ⟨214⟩, ⟨16152⟩⟩
def transferEvent : Nat := 19147
def frameStart : Nat := 19082
def rule : BoundRule := .product (.predecessor 0 19145 .coefficient) (.predecessor 1 19146 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19145 .coefficient)
      LeftAuthority19143.bound (LeftAuthority19143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19146 .coefficient)
      LeftBound19141.bound (LeftBound19141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority19143.bound LeftBound19141.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19143.bound, LeftBound19141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority19143.actual selector witness) * (LeftBound19141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19147

namespace LeftBound19155
def owner : Owner := ⟨.program ⟨214⟩, ⟨16153⟩⟩
def transferEvent : Nat := 19155
def frameStart : Nat := 19082
def rule : BoundRule := .sum [.predecessor 0 19153 .coefficient, .predecessor 1 19154 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19153 .coefficient)
      LeftAuthority19151.bound (LeftAuthority19151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19154 .coefficient)
      LeftBound19147.bound (LeftBound19147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19147.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19151.bound, LeftBound19147.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19151.bound, LeftBound19147.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19151.actual selector witness, LeftBound19147.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19155

namespace LeftBound19159
def owner : Owner := ⟨.program ⟨214⟩, ⟨28129⟩⟩
def transferEvent : Nat := 19159
def frameStart : Nat := 19082
def rule : BoundRule := .product (.predecessor 0 19157 .coefficient) (.predecessor 1 19158 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19157 .coefficient)
      LeftBound19155.bound (LeftBound19155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19158 .coefficient)
      LeftAuthority19132.bound (LeftAuthority19132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19133RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19132.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19132.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound19155.bound LeftAuthority19132.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound19155.bound, LeftAuthority19132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound19155.actual selector witness) * (LeftAuthority19132.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19159

namespace LeftBound19170
def owner : Owner := ⟨.program ⟨214⟩, ⟨18068⟩⟩
def transferEvent : Nat := 19170
def frameStart : Nat := 19082
def rule : BoundRule := .product (.predecessor 0 19168 .coefficient) (.predecessor 1 19169 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19168 .coefficient)
      LeftAuthority19143.bound (LeftAuthority19143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19143.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19143.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19169 .coefficient)
      LeftAuthority19166.bound (LeftAuthority19166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19167RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19166.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19166.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority19143.bound LeftAuthority19166.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19143.bound, LeftAuthority19166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority19143.actual selector witness) * (LeftAuthority19166.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound19170

namespace LeftBound19178
def owner : Owner := ⟨.program ⟨214⟩, ⟨18069⟩⟩
def transferEvent : Nat := 19178
def frameStart : Nat := 19082
def rule : BoundRule := .sum [.predecessor 0 19176 .coefficient, .predecessor 1 19177 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 19176 .coefficient)
      LeftAuthority19174.bound (LeftAuthority19174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority19174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority19174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 19177 .coefficient)
      LeftBound19170.bound (LeftBound19170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events074.exact19172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound19170.bound, RecordedBoundRefines] <;> decide)
      (LeftBound19170.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority19174.bound, LeftBound19170.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority19174.bound, LeftBound19170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority19174.actual selector witness, LeftBound19170.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound19178

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
