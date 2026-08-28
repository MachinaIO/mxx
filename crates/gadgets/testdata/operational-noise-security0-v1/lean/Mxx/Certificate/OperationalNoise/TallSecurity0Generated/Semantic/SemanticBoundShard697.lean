import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard641
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard695
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard696

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound101133
def owner : Owner := ⟨.program ⟨214⟩, ⟨25054⟩⟩
def transferEvent : Nat := 101133
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 101127 .summary, .result 100965 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101127 .summary)
      LeftBound100977.bound (LeftBound100977.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19160⟩⟩) (rawTerms := some (Proof.Events395.exact101127RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100977.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100965 .summary)
      LeftBound100960.bound (LeftBound100960.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25053⟩⟩) (rawTerms := some (Proof.Events394.exact100965RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound100960.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound100977.bound, LeftBound100960.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound100977.bound, LeftBound100960.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound100977.actual selector witness, LeftBound100960.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101133

namespace LeftBound101137
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def transferEvent : Nat := 101137
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101135 .coefficient) (.predecessor 1 101136 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101135 .coefficient)
      LeftBound101130.bound (LeftBound101130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101136 .coefficient)
      LeftAuthority100880.bound (LeftAuthority100880.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events394.exact100881RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100880.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101130.bound LeftAuthority100880.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101130.bound, LeftAuthority100880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101130.actual selector witness) * (LeftAuthority100880.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101137

namespace LeftBound101138
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def transferEvent : Nat := 101138
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26746⟩⟩]⟩ [⟨.result 100881 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 100881 .coefficient)
      LeftAuthority100880.bound (LeftAuthority100880.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26746⟩⟩) (rawTerms := some (Proof.Events394.exact100881RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority100880.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority100880.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority100880.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority100880.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority100880.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101138

namespace LeftBound101139
def owner : Owner := ⟨.program ⟨214⟩, ⟨26748⟩⟩
def transferEvent : Nat := 101139
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 101134 .summary) (.transfer 101138) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101134 .summary)
      LeftBound101133.bound (LeftBound101133.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25054⟩⟩) (rawTerms := some (Proof.Events395.exact101134RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound101133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101138)
      LeftBound101138.bound (LeftBound101138.actual selector witness) := by
  exact .transfer (LeftBound101138.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101133.bound LeftBound101138.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101133.bound, LeftBound101138.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101133.actual selector witness) * (LeftBound101138.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101139

namespace LeftBound101150
def owner : Owner := ⟨.program ⟨214⟩, ⟨20671⟩⟩
def transferEvent : Nat := 101150
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 101148 .coefficient) (.value (.predecessor 1 101149 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101148 .coefficient)
      LeftAuthority101146.bound (LeftAuthority101146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101146.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101149 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority101146.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101146.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101146.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound101150

namespace LeftBound101154
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def transferEvent : Nat := 101154
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 101152 .coefficient) (.predecessor 1 101153 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101152 .coefficient)
      LeftBound94459.bound (LeftBound94459.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound94459.bound, RecordedBoundRefines] <;> decide)
      (LeftBound94459.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101153 .coefficient)
      LeftBound101150.bound (LeftBound101150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101151RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101150.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94459.bound LeftBound101150.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94459.bound, LeftBound101150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94459.actual selector witness) * (LeftBound101150.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101154

namespace LeftBound101155
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def transferEvent : Nat := 101155
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20669⟩⟩]⟩ [⟨.result 101147 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 101147 .coefficient)
      LeftAuthority101146.bound (LeftAuthority101146.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20669⟩⟩) (rawTerms := some (Proof.Events395.exact101147RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101146.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority101146.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority101146.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound101155

namespace LeftBound101156
def owner : Owner := ⟨.program ⟨214⟩, ⟨20672⟩⟩
def transferEvent : Nat := 101156
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 94462 .summary) (.transfer 101155) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 94462 .summary)
      LeftBound94460.bound (LeftBound94460.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5509⟩⟩) (rawTerms := some (Proof.Events368.exact94462RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound94460.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 101155)
      LeftBound101155.bound (LeftBound101155.actual selector witness) := by
  exact .transfer (LeftBound101155.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound94460.bound LeftBound101155.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound94460.bound, LeftBound101155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound94460.actual selector witness) * (LeftBound101155.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101156

namespace LeftBound101227
def owner : Owner := ⟨.program ⟨214⟩, ⟨15105⟩⟩
def transferEvent : Nat := 101227
def frameStart : Nat := 101200
def rule : BoundRule := .identity (.predecessor 0 101226 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101226 .coefficient)
      LeftAuthority101224.bound (LeftAuthority101224.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101225RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101224.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101224.derived selector witness)

def rawBound : CoeffClass := LeftAuthority101224.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101224.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority101224.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101227

namespace LeftBound101244
def owner : Owner := ⟨.program ⟨214⟩, ⟨15146⟩⟩
def transferEvent : Nat := 101244
def frameStart : Nat := 101200
def rule : BoundRule := .sum [.predecessor 0 101242 .coefficient, .predecessor 1 101243 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101242 .coefficient)
      LeftBound101227.bound (LeftBound101227.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101227.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101243 .coefficient)
      LeftAuthority101240.bound (LeftAuthority101240.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority101240.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound101227.bound, LeftAuthority101240.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101227.bound, LeftAuthority101240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound101227.actual selector witness, LeftAuthority101240.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101244

namespace LeftBound101247
def owner : Owner := ⟨.program ⟨214⟩, ⟨15147⟩⟩
def transferEvent : Nat := 101247
def frameStart : Nat := 101200
def rule : BoundRule := .identity (.predecessor 0 101246 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101246 .coefficient)
      LeftBound101244.bound (LeftBound101244.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound101244.derived selector witness)

def rawBound : CoeffClass := LeftBound101244.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101244.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound101244.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound101247

namespace LeftBound101253
def owner : Owner := ⟨.program ⟨214⟩, ⟨15148⟩⟩
def transferEvent : Nat := 101253
def frameStart : Nat := 101200
def rule : BoundRule := .product (.predecessor 0 101251 .coefficient) (.predecessor 1 101252 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101251 .coefficient)
      LeftAuthority101249.bound (LeftAuthority101249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101252 .coefficient)
      LeftBound101247.bound (LeftBound101247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101247.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority101249.bound LeftBound101247.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101249.bound, LeftBound101247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority101249.actual selector witness) * (LeftBound101247.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101253

namespace LeftBound101261
def owner : Owner := ⟨.program ⟨214⟩, ⟨15149⟩⟩
def transferEvent : Nat := 101261
def frameStart : Nat := 101200
def rule : BoundRule := .sum [.predecessor 0 101259 .coefficient, .predecessor 1 101260 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101259 .coefficient)
      LeftAuthority101257.bound (LeftAuthority101257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101258RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101257.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101257.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101260 .coefficient)
      LeftBound101253.bound (LeftBound101253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101255RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101253.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101253.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101257.bound, LeftBound101253.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101257.bound, LeftBound101253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101257.actual selector witness, LeftBound101253.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101261

namespace LeftBound101265
def owner : Owner := ⟨.program ⟨214⟩, ⟨26747⟩⟩
def transferEvent : Nat := 101265
def frameStart : Nat := 101200
def rule : BoundRule := .product (.predecessor 0 101263 .coefficient) (.predecessor 1 101264 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101263 .coefficient)
      LeftBound101261.bound (LeftBound101261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101261.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101264 .coefficient)
      LeftAuthority101238.bound (LeftAuthority101238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101238.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101238.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound101261.bound LeftAuthority101238.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound101261.bound, LeftAuthority101238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound101261.actual selector witness) * (LeftAuthority101238.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101265

namespace LeftBound101276
def owner : Owner := ⟨.program ⟨214⟩, ⟨15358⟩⟩
def transferEvent : Nat := 101276
def frameStart : Nat := 101200
def rule : BoundRule := .product (.predecessor 0 101274 .coefficient) (.predecessor 1 101275 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101274 .coefficient)
      LeftAuthority101249.bound (LeftAuthority101249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101275 .coefficient)
      LeftAuthority101272.bound (LeftAuthority101272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101272.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101272.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority101249.bound LeftAuthority101272.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101249.bound, LeftAuthority101272.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority101249.actual selector witness) * (LeftAuthority101272.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound101276

namespace LeftBound101284
def owner : Owner := ⟨.program ⟨214⟩, ⟨15359⟩⟩
def transferEvent : Nat := 101284
def frameStart : Nat := 101200
def rule : BoundRule := .sum [.predecessor 0 101282 .coefficient, .predecessor 1 101283 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 101282 .coefficient)
      LeftAuthority101280.bound (LeftAuthority101280.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101281RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority101280.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority101280.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 101283 .coefficient)
      LeftBound101276.bound (LeftBound101276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events395.exact101278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound101276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound101276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority101280.bound, LeftBound101276.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority101280.bound, LeftBound101276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority101280.actual selector witness, LeftBound101276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound101284

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
