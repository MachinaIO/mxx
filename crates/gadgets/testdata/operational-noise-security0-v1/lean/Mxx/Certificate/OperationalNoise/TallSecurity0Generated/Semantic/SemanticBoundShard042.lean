import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard030
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard041

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound8046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12801⟩⟩
def transferEvent : Nat := 8046
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 8041 .summary, .result 7998 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8041 .summary)
      LeftBound8036.bound (LeftBound8036.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10054⟩⟩) (rawTerms := some (Proof.Events031.exact8041RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8036.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7998 .summary)
      LeftBound7995.bound (LeftBound7995.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12800⟩⟩) (rawTerms := some (Proof.Events031.exact7998RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound7995.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8036.bound, LeftBound7995.bound]
def bound : CoeffClass := .finite ⟨95458688, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8036.bound, LeftBound7995.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8036.actual selector witness, LeftBound7995.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8046

namespace LeftBound8050
def owner : Owner := ⟨.program ⟨214⟩, ⟨25548⟩⟩
def transferEvent : Nat := 8050
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8048 .coefficient) (.predecessor 1 8049 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8048 .coefficient)
      LeftBound8044.bound (LeftBound8044.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8047RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8044.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8044.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8049 .coefficient)
      LeftAuthority7963.bound (LeftAuthority7963.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact7964RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7963.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8044.bound LeftAuthority7963.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8044.bound, LeftAuthority7963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8044.actual selector witness) * (LeftAuthority7963.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8050

namespace LeftBound8051
def owner : Owner := ⟨.program ⟨214⟩, ⟨25548⟩⟩
def transferEvent : Nat := 8051
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25547⟩⟩]⟩ [⟨.result 7964 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 7964 .coefficient)
      LeftAuthority7963.bound (LeftAuthority7963.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25547⟩⟩) (rawTerms := some (Proof.Events031.exact7964RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7963.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7963.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7963.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority7963.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8051

namespace LeftBound8052
def owner : Owner := ⟨.program ⟨214⟩, ⟨25548⟩⟩
def transferEvent : Nat := 8052
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 8047 .summary) (.transfer 8051) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8047 .summary)
      LeftBound8046.bound (LeftBound8046.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12801⟩⟩) (rawTerms := some (Proof.Events031.exact8047RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound8046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8051)
      LeftBound8051.bound (LeftBound8051.actual selector witness) := by
  exact .transfer (LeftBound8051.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8046.bound LeftBound8051.bound
def bound : CoeffClass := .finite ⟨350334912299008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8046.bound, LeftBound8051.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8046.actual selector witness) * (LeftBound8051.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8052

namespace LeftBound8063
def owner : Owner := ⟨.program ⟨214⟩, ⟨20050⟩⟩
def transferEvent : Nat := 8063
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 8061 .coefficient) (.value (.predecessor 1 8062 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8061 .coefficient)
      LeftAuthority8059.bound (LeftAuthority8059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8062 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority8059.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8059.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8059.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound8063

namespace LeftBound8067
def owner : Owner := ⟨.program ⟨214⟩, ⟨20051⟩⟩
def transferEvent : Nat := 8067
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 8065 .coefficient) (.predecessor 1 8066 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8065 .coefficient)
      LeftBound6558.bound (LeftBound6558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8066 .coefficient)
      LeftBound8063.bound (LeftBound8063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8063.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6558.bound LeftBound8063.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6558.bound, LeftBound8063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6558.actual selector witness) * (LeftBound8063.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8067

namespace LeftBound8068
def owner : Owner := ⟨.program ⟨214⟩, ⟨20051⟩⟩
def transferEvent : Nat := 8068
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20048⟩⟩]⟩ [⟨.result 8060 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 8060 .coefficient)
      LeftAuthority8059.bound (LeftAuthority8059.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20048⟩⟩) (rawTerms := some (Proof.Events031.exact8060RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8059.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority8059.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8059.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound8068

namespace LeftBound8069
def owner : Owner := ⟨.program ⟨214⟩, ⟨20051⟩⟩
def transferEvent : Nat := 8069
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 6561 .summary) (.transfer 8068) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6561 .summary)
      LeftBound6559.bound (LeftBound6559.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5565⟩⟩) (rawTerms := some (Proof.Events025.exact6561RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound6559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 8068)
      LeftBound8068.bound (LeftBound8068.actual selector witness) := by
  exact .transfer (LeftBound8068.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound6559.bound LeftBound8068.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6559.bound, LeftBound8068.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound6559.actual selector witness) * (LeftBound8068.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8069

namespace LeftBound8148
def owner : Owner := ⟨.program ⟨214⟩, ⟨12795⟩⟩
def transferEvent : Nat := 8148
def frameStart : Nat := 8119
def rule : BoundRule := .product (.predecessor 0 8146 .coefficient) (.predecessor 1 8147 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8146 .coefficient)
      LeftAuthority8144.bound (LeftAuthority8144.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8144.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8144.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8147 .coefficient)
      LeftAuthority8141.bound (LeftAuthority8141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8141.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8141.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority8144.bound LeftAuthority8141.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8144.bound, LeftAuthority8141.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority8144.actual selector witness) * (LeftAuthority8141.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8148

namespace LeftBound8152
def owner : Owner := ⟨.program ⟨214⟩, ⟨12796⟩⟩
def transferEvent : Nat := 8152
def frameStart : Nat := 8119
def rule : BoundRule := .identity (.predecessor 0 8151 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8151 .coefficient)
      LeftBound8148.bound (LeftBound8148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8148.derived selector witness)

def rawBound : CoeffClass := LeftBound8148.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound8148.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8152

namespace LeftBound8169
def owner : Owner := ⟨.program ⟨214⟩, ⟨12874⟩⟩
def transferEvent : Nat := 8169
def frameStart : Nat := 8119
def rule : BoundRule := .sum [.predecessor 0 8167 .coefficient, .predecessor 1 8168 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8167 .coefficient)
      LeftBound8152.bound (LeftBound8152.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound8152.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8168 .coefficient)
      LeftAuthority8165.bound (LeftAuthority8165.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority8165.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound8152.bound, LeftAuthority8165.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8152.bound, LeftAuthority8165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound8152.actual selector witness, LeftAuthority8165.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound8169

namespace LeftBound8172
def owner : Owner := ⟨.program ⟨214⟩, ⟨12875⟩⟩
def transferEvent : Nat := 8172
def frameStart : Nat := 8119
def rule : BoundRule := .identity (.predecessor 0 8171 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8171 .coefficient)
      LeftBound8169.bound (LeftBound8169.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound8169.derived selector witness)

def rawBound : CoeffClass := LeftBound8169.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound8169.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8172

namespace LeftBound8178
def owner : Owner := ⟨.program ⟨214⟩, ⟨12876⟩⟩
def transferEvent : Nat := 8178
def frameStart : Nat := 8119
def rule : BoundRule := .product (.predecessor 0 8176 .coefficient) (.predecessor 1 8177 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8176 .coefficient)
      LeftAuthority8174.bound (LeftAuthority8174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8177 .coefficient)
      LeftBound8172.bound (LeftBound8172.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8173RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8172.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8172.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority8174.bound LeftBound8172.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8174.bound, LeftBound8172.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority8174.actual selector witness) * (LeftBound8172.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8178

namespace LeftBound8194
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 8194
def frameStart : Nat := 8119
def rule : BoundRule := .scale (.predecessor 0 8192 .coefficient) (.value (.predecessor 1 8193 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8192 .coefficient)
      LeftAuthority8190.bound (LeftAuthority8190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8191RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8190.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8193 .coefficient)
      LeftAuthority8181.bound (LeftAuthority8181.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority8181.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority8190.bound LeftAuthority8181.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8190.bound, LeftAuthority8181.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority8190.actual selector witness) * (LeftAuthority8181.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound8194

namespace LeftBound8197
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 8197
def frameStart : Nat := 8119
def rule : BoundRule := .identity (.predecessor 0 8196 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8196 .coefficient)
      LeftAuthority8184.bound (LeftAuthority8184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events031.exact8185RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority8184.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority8184.derived selector witness)

def rawBound : CoeffClass := LeftAuthority8184.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority8184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority8184.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound8197

namespace LeftBound8201
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 8201
def frameStart : Nat := 8119
def rule : BoundRule := .product (.predecessor 0 8199 .coefficient) (.predecessor 1 8200 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 8199 .coefficient)
      LeftBound8197.bound (LeftBound8197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 8200 .coefficient)
      LeftBound8194.bound (LeftBound8194.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events032.exact8195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound8194.bound, RecordedBoundRefines] <;> decide)
      (LeftBound8194.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound8197.bound LeftBound8194.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound8197.bound, LeftBound8194.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound8197.actual selector witness) * (LeftBound8194.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound8201

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
