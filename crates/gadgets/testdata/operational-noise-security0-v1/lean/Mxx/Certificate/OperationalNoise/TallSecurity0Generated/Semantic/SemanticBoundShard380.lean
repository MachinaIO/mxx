import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard340

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56063
def owner : Owner := ⟨.program ⟨214⟩, ⟨19462⟩⟩
def transferEvent : Nat := 56063
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 56061 .coefficient) (.value (.predecessor 1 56062 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56061 .coefficient)
      LeftAuthority56059.bound (LeftAuthority56059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events218.exact56060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56062 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56059.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56059.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56059.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56063

namespace LeftBound56067
def owner : Owner := ⟨.program ⟨214⟩, ⟨19463⟩⟩
def transferEvent : Nat := 56067
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 56065 .coefficient) (.predecessor 1 56066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56065 .coefficient)
      LeftBound50759.bound (LeftBound50759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound50759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound50759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56066 .coefficient)
      LeftBound56063.bound (LeftBound56063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50759.bound LeftBound56063.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50759.bound, LeftBound56063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50759.actual selector witness) * (LeftBound56063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56067

namespace LeftBound56068
def owner : Owner := ⟨.program ⟨214⟩, ⟨19463⟩⟩
def transferEvent : Nat := 56068
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19460⟩⟩]⟩ [⟨.result 56060 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 56060 .coefficient)
      LeftAuthority56059.bound (LeftAuthority56059.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19460⟩⟩) (rawTerms := some (Proof.Events218.exact56060RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56059.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority56059.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56059.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound56068

namespace LeftBound56069
def owner : Owner := ⟨.program ⟨214⟩, ⟨19463⟩⟩
def transferEvent : Nat := 56069
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 50762 .summary) (.transfer 56068) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 50762 .summary)
      LeftBound50760.bound (LeftBound50760.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5547⟩⟩) (rawTerms := some (Proof.Events198.exact50762RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound50760.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 56068)
      LeftBound56068.bound (LeftBound56068.actual selector witness) := by
  exact .transfer (LeftBound56068.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound50760.bound LeftBound56068.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound50760.bound, LeftBound56068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound50760.actual selector witness) * (LeftBound56068.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56069

namespace LeftBound56148
def owner : Owner := ⟨.program ⟨214⟩, ⟨14000⟩⟩
def transferEvent : Nat := 56148
def frameStart : Nat := 56119
def rule : BoundRule := .product (.predecessor 0 56146 .coefficient) (.predecessor 1 56147 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56146 .coefficient)
      LeftAuthority56144.bound (LeftAuthority56144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56147 .coefficient)
      LeftAuthority56141.bound (LeftAuthority56141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56144.bound LeftAuthority56141.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56144.bound, LeftAuthority56141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority56144.actual selector witness) * (LeftAuthority56141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56148

namespace LeftBound56152
def owner : Owner := ⟨.program ⟨214⟩, ⟨14001⟩⟩
def transferEvent : Nat := 56152
def frameStart : Nat := 56119
def rule : BoundRule := .identity (.predecessor 0 56151 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56151 .coefficient)
      LeftBound56148.bound (LeftBound56148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56148.derived selector witness)

def rawBound : CoeffClass := LeftBound56148.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56148.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56152

namespace LeftBound56169
def owner : Owner := ⟨.program ⟨214⟩, ⟨14101⟩⟩
def transferEvent : Nat := 56169
def frameStart : Nat := 56119
def rule : BoundRule := .sum [.predecessor 0 56167 .coefficient, .predecessor 1 56168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56167 .coefficient)
      LeftBound56152.bound (LeftBound56152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56168 .coefficient)
      LeftAuthority56165.bound (LeftAuthority56165.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56152.bound, LeftAuthority56165.bound]
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56152.bound, LeftAuthority56165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56152.actual selector witness, LeftAuthority56165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56169

namespace LeftBound56172
def owner : Owner := ⟨.program ⟨214⟩, ⟨14102⟩⟩
def transferEvent : Nat := 56172
def frameStart : Nat := 56119
def rule : BoundRule := .identity (.predecessor 0 56171 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56171 .coefficient)
      LeftBound56169.bound (LeftBound56169.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56169.derived selector witness)

def rawBound : CoeffClass := LeftBound56169.bound
def bound : CoeffClass := .finite ⟨256, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound56169.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56172

namespace LeftBound56178
def owner : Owner := ⟨.program ⟨214⟩, ⟨14103⟩⟩
def transferEvent : Nat := 56178
def frameStart : Nat := 56119
def rule : BoundRule := .product (.predecessor 0 56176 .coefficient) (.predecessor 1 56177 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56176 .coefficient)
      LeftAuthority56174.bound (LeftAuthority56174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56177 .coefficient)
      LeftBound56172.bound (LeftBound56172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority56174.bound LeftBound56172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56174.bound, LeftBound56172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority56174.actual selector witness) * (LeftBound56172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56178

namespace LeftBound56194
def owner : Owner := ⟨.program ⟨214⟩, ⟨7850⟩⟩
def transferEvent : Nat := 56194
def frameStart : Nat := 56119
def rule : BoundRule := .scale (.predecessor 0 56192 .coefficient) (.value (.predecessor 1 56193 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56192 .coefficient)
      LeftAuthority56190.bound (LeftAuthority56190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56193 .coefficient)
      LeftAuthority56181.bound (LeftAuthority56181.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56181.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority56190.bound LeftAuthority56181.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56190.bound, LeftAuthority56181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority56190.actual selector witness) * (LeftAuthority56181.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound56194

namespace LeftBound56197
def owner : Owner := ⟨.program ⟨214⟩, ⟨6758⟩⟩
def transferEvent : Nat := 56197
def frameStart : Nat := 56119
def rule : BoundRule := .identity (.predecessor 0 56196 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56196 .coefficient)
      LeftAuthority56184.bound (LeftAuthority56184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56184.derived selector witness)

def rawBound : CoeffClass := LeftAuthority56184.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority56184.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56197

namespace LeftBound56201
def owner : Owner := ⟨.program ⟨214⟩, ⟨7851⟩⟩
def transferEvent : Nat := 56201
def frameStart : Nat := 56119
def rule : BoundRule := .product (.predecessor 0 56199 .coefficient) (.predecessor 1 56200 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56199 .coefficient)
      LeftBound56197.bound (LeftBound56197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56200 .coefficient)
      LeftBound56194.bound (LeftBound56194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56197.bound LeftBound56194.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56197.bound, LeftBound56194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56197.actual selector witness) * (LeftBound56194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56201

namespace LeftBound56206
def owner : Owner := ⟨.program ⟨214⟩, ⟨14104⟩⟩
def transferEvent : Nat := 56206
def frameStart : Nat := 56119
def rule : BoundRule := .sum [.predecessor 0 56204 .coefficient, .predecessor 1 56205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56204 .coefficient)
      LeftBound56201.bound (LeftBound56201.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56203RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56201.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56201.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56205 .coefficient)
      LeftBound56178.bound (LeftBound56178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56201.bound, LeftBound56178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56201.bound, LeftBound56178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound56201.actual selector witness, LeftBound56178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56206

namespace LeftBound56210
def owner : Owner := ⟨.program ⟨214⟩, ⟨25997⟩⟩
def transferEvent : Nat := 56210
def frameStart : Nat := 56119
def rule : BoundRule := .product (.predecessor 0 56208 .coefficient) (.predecessor 1 56209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56208 .coefficient)
      LeftBound56206.bound (LeftBound56206.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56206.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56209 .coefficient)
      LeftAuthority56163.bound (LeftAuthority56163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56163.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56163.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound56206.bound LeftAuthority56163.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56206.bound, LeftAuthority56163.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound56206.actual selector witness) * (LeftAuthority56163.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56210

namespace LeftBound56221
def owner : Owner := ⟨.program ⟨214⟩, ⟨15827⟩⟩
def transferEvent : Nat := 56221
def frameStart : Nat := 56119
def rule : BoundRule := .product (.predecessor 0 56219 .coefficient) (.predecessor 1 56220 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56219 .coefficient)
      LeftAuthority56174.bound (LeftAuthority56174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56220 .coefficient)
      LeftAuthority56217.bound (LeftAuthority56217.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56217.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56217.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority56174.bound LeftAuthority56217.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56174.bound, LeftAuthority56217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority56174.actual selector witness) * (LeftAuthority56217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56221

namespace LeftBound56229
def owner : Owner := ⟨.program ⟨214⟩, ⟨15828⟩⟩
def transferEvent : Nat := 56229
def frameStart : Nat := 56119
def rule : BoundRule := .sum [.predecessor 0 56227 .coefficient, .predecessor 1 56228 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 56227 .coefficient)
      LeftAuthority56225.bound (LeftAuthority56225.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56226RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56225.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56225.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 56228 .coefficient)
      LeftBound56221.bound (LeftBound56221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56225.bound, LeftBound56221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56225.bound, LeftBound56221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority56225.actual selector witness, LeftBound56221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56229

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
