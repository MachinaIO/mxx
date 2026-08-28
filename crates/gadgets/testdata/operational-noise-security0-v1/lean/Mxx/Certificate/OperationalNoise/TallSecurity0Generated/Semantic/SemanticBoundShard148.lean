import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard147

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound22945
def owner : Owner := ⟨.program ⟨214⟩, ⟨25543⟩⟩
def transferEvent : Nat := 22945
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25542⟩⟩]⟩ [⟨.result 22877 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22877 .coefficient)
      LeftAuthority22876.bound (LeftAuthority22876.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25542⟩⟩) (rawTerms := some (Proof.Events089.exact22877RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22876.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22876.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22876.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22876.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22876.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22945

namespace LeftBound22946
def owner : Owner := ⟨.program ⟨214⟩, ⟨25543⟩⟩
def transferEvent : Nat := 22946
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 22941 .summary) (.transfer 22945) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22941 .summary)
      LeftBound22940.bound (LeftBound22940.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨12793⟩⟩) (rawTerms := some (Proof.Events089.exact22941RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound22940.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22945)
      LeftBound22945.bound (LeftBound22945.actual selector witness) := by
  exact .transfer (LeftBound22945.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound22940.bound LeftBound22945.bound
def bound : CoeffClass := .finite ⟨350334912299008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound22940.bound, LeftBound22945.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound22940.actual selector witness) * (LeftBound22945.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22946

namespace LeftBound22957
def owner : Owner := ⟨.program ⟨214⟩, ⟨20046⟩⟩
def transferEvent : Nat := 22957
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 22955 .coefficient) (.value (.predecessor 1 22956 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22955 .coefficient)
      LeftAuthority22953.bound (LeftAuthority22953.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22954RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22953.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22956 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority22953.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22953.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22953.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound22957

namespace LeftBound22961
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def transferEvent : Nat := 22961
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 22959 .coefficient) (.predecessor 1 22960 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 22959 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 22960 .coefficient)
      LeftBound22957.bound (LeftBound22957.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact22958RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound22957.bound, RecordedBoundRefines] <;> decide)
      (LeftBound22957.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound22957.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound22957.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound22957.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22961

namespace LeftBound22962
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def transferEvent : Nat := 22962
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20044⟩⟩]⟩ [⟨.result 22954 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 22954 .coefficient)
      LeftAuthority22953.bound (LeftAuthority22953.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20044⟩⟩) (rawTerms := some (Proof.Events089.exact22954RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority22953.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority22953.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority22953.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority22953.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority22953.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound22962

namespace LeftBound22963
def owner : Owner := ⟨.program ⟨214⟩, ⟨20047⟩⟩
def transferEvent : Nat := 22963
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 22962) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 22962)
      LeftBound22962.bound (LeftBound22962.actual selector witness) := by
  exact .transfer (LeftBound22962.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound22962.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound22962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound22962.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound22963

namespace LeftBound23042
def owner : Owner := ⟨.program ⟨214⟩, ⟨12787⟩⟩
def transferEvent : Nat := 23042
def frameStart : Nat := 23013
def rule : BoundRule := .product (.predecessor 0 23040 .coefficient) (.predecessor 1 23041 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23040 .coefficient)
      LeftAuthority23038.bound (LeftAuthority23038.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact23039RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23038.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23038.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23041 .coefficient)
      LeftAuthority23035.bound (LeftAuthority23035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events089.exact23036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23035.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23035.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority23038.bound LeftAuthority23035.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23038.bound, LeftAuthority23035.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority23038.actual selector witness) * (LeftAuthority23035.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23042

namespace LeftBound23046
def owner : Owner := ⟨.program ⟨214⟩, ⟨12788⟩⟩
def transferEvent : Nat := 23046
def frameStart : Nat := 23013
def rule : BoundRule := .identity (.predecessor 0 23045 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23045 .coefficient)
      LeftBound23042.bound (LeftBound23042.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23042.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23042.derived selector witness)

def rawBound : CoeffClass := LeftBound23042.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23042.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23042.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23046

namespace LeftBound23063
def owner : Owner := ⟨.program ⟨214⟩, ⟨12870⟩⟩
def transferEvent : Nat := 23063
def frameStart : Nat := 23013
def rule : BoundRule := .sum [.predecessor 0 23061 .coefficient, .predecessor 1 23062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23061 .coefficient)
      LeftBound23046.bound (LeftBound23046.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23046.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23062 .coefficient)
      LeftAuthority23059.bound (LeftAuthority23059.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23046.bound, LeftAuthority23059.bound]
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23046.bound, LeftAuthority23059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23046.actual selector witness, LeftAuthority23059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23063

namespace LeftBound23066
def owner : Owner := ⟨.program ⟨214⟩, ⟨12871⟩⟩
def transferEvent : Nat := 23066
def frameStart : Nat := 23013
def rule : BoundRule := .identity (.predecessor 0 23065 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23065 .coefficient)
      LeftBound23063.bound (LeftBound23063.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound23063.derived selector witness)

def rawBound : CoeffClass := LeftBound23063.bound
def bound : CoeffClass := .finite ⟨2116, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23063.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound23063.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23066

namespace LeftBound23072
def owner : Owner := ⟨.program ⟨214⟩, ⟨12872⟩⟩
def transferEvent : Nat := 23072
def frameStart : Nat := 23013
def rule : BoundRule := .product (.predecessor 0 23070 .coefficient) (.predecessor 1 23071 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23070 .coefficient)
      LeftAuthority23068.bound (LeftAuthority23068.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23069RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23068.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23068.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23071 .coefficient)
      LeftBound23066.bound (LeftBound23066.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23067RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23066.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23066.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority23068.bound LeftBound23066.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23068.bound, LeftBound23066.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority23068.actual selector witness) * (LeftBound23066.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23072

namespace LeftBound23088
def owner : Owner := ⟨.program ⟨214⟩, ⟨7874⟩⟩
def transferEvent : Nat := 23088
def frameStart : Nat := 23013
def rule : BoundRule := .scale (.predecessor 0 23086 .coefficient) (.value (.predecessor 1 23087 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23086 .coefficient)
      LeftAuthority23084.bound (LeftAuthority23084.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23085RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23084.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23084.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23087 .coefficient)
      LeftAuthority23075.bound (LeftAuthority23075.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority23075.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority23084.bound LeftAuthority23075.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23084.bound, LeftAuthority23075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority23084.actual selector witness) * (LeftAuthority23075.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound23088

namespace LeftBound23091
def owner : Owner := ⟨.program ⟨214⟩, ⟨6767⟩⟩
def transferEvent : Nat := 23091
def frameStart : Nat := 23013
def rule : BoundRule := .identity (.predecessor 0 23090 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23090 .coefficient)
      LeftAuthority23078.bound (LeftAuthority23078.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23079RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23078.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23078.derived selector witness)

def rawBound : CoeffClass := LeftAuthority23078.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority23078.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority23078.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound23091

namespace LeftBound23095
def owner : Owner := ⟨.program ⟨214⟩, ⟨7875⟩⟩
def transferEvent : Nat := 23095
def frameStart : Nat := 23013
def rule : BoundRule := .product (.predecessor 0 23093 .coefficient) (.predecessor 1 23094 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23093 .coefficient)
      LeftBound23091.bound (LeftBound23091.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23091.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23091.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23094 .coefficient)
      LeftBound23088.bound (LeftBound23088.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23089RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23088.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23088.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23091.bound LeftBound23088.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23091.bound, LeftBound23088.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23091.actual selector witness) * (LeftBound23088.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23095

namespace LeftBound23100
def owner : Owner := ⟨.program ⟨214⟩, ⟨12873⟩⟩
def transferEvent : Nat := 23100
def frameStart : Nat := 23013
def rule : BoundRule := .sum [.predecessor 0 23098 .coefficient, .predecessor 1 23099 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23098 .coefficient)
      LeftBound23095.bound (LeftBound23095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23095.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23099 .coefficient)
      LeftBound23072.bound (LeftBound23072.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23074RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23072.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23072.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound23095.bound, LeftBound23072.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23095.bound, LeftBound23072.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound23095.actual selector witness, LeftBound23072.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound23100

namespace LeftBound23104
def owner : Owner := ⟨.program ⟨214⟩, ⟨25545⟩⟩
def transferEvent : Nat := 23104
def frameStart : Nat := 23013
def rule : BoundRule := .product (.predecessor 0 23102 .coefficient) (.predecessor 1 23103 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 23102 .coefficient)
      LeftBound23100.bound (LeftBound23100.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23101RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound23100.bound, RecordedBoundRefines] <;> decide)
      (LeftBound23100.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 23103 .coefficient)
      LeftAuthority23057.bound (LeftAuthority23057.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events090.exact23058RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority23057.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority23057.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound23100.bound LeftAuthority23057.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound23100.bound, LeftAuthority23057.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound23100.actual selector witness) * (LeftAuthority23057.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound23104

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
