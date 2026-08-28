import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard473

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound69711
def owner : Owner := ⟨.program ⟨214⟩, ⟨26139⟩⟩
def transferEvent : Nat := 69711
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69709 .coefficient) (.predecessor 1 69710 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69709 .coefficient)
      LeftBound69705.bound (LeftBound69705.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69705.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69705.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69710 .coefficient)
      LeftAuthority69643.bound (LeftAuthority69643.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69644RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69643.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69705.bound LeftAuthority69643.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69705.bound, LeftAuthority69643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69705.actual selector witness) * (LeftAuthority69643.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69711

namespace LeftBound69712
def owner : Owner := ⟨.program ⟨214⟩, ⟨26139⟩⟩
def transferEvent : Nat := 69712
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26138⟩⟩]⟩ [⟨.result 69644 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69644 .coefficient)
      LeftAuthority69643.bound (LeftAuthority69643.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26138⟩⟩) (rawTerms := some (Proof.Events272.exact69644RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69643.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69643.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69643.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69643.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69643.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69712

namespace LeftBound69713
def owner : Owner := ⟨.program ⟨214⟩, ⟨26139⟩⟩
def transferEvent : Nat := 69713
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 69708 .summary) (.transfer 69712) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69708 .summary)
      LeftBound69707.bound (LeftBound69707.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨14423⟩⟩) (rawTerms := some (Proof.Events272.exact69708RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound69707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69712)
      LeftBound69712.bound (LeftBound69712.actual selector witness) := by
  exact .transfer (LeftBound69712.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69707.bound LeftBound69712.bound
def bound : CoeffClass := .finite ⟨350261629419520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69707.bound, LeftBound69712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69707.actual selector witness) * (LeftBound69712.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69713

namespace LeftBound69724
def owner : Owner := ⟨.program ⟨214⟩, ⟨19598⟩⟩
def transferEvent : Nat := 69724
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 69722 .coefficient) (.value (.predecessor 1 69723 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69722 .coefficient)
      LeftAuthority69720.bound (LeftAuthority69720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69720.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69723 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority69720.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69720.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69720.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69724

namespace LeftBound69728
def owner : Owner := ⟨.program ⟨214⟩, ⟨19599⟩⟩
def transferEvent : Nat := 69728
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 69726 .coefficient) (.predecessor 1 69727 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69726 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69727 .coefficient)
      LeftBound69724.bound (LeftBound69724.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69725RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69724.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69724.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound69724.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound69724.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound69724.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69728

namespace LeftBound69729
def owner : Owner := ⟨.program ⟨214⟩, ⟨19599⟩⟩
def transferEvent : Nat := 69729
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19596⟩⟩]⟩ [⟨.result 69721 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 69721 .coefficient)
      LeftAuthority69720.bound (LeftAuthority69720.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19596⟩⟩) (rawTerms := some (Proof.Events272.exact69721RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69720.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority69720.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69720.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound69729

namespace LeftBound69730
def owner : Owner := ⟨.program ⟨214⟩, ⟨19599⟩⟩
def transferEvent : Nat := 69730
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 69729) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 69729)
      LeftBound69729.bound (LeftBound69729.actual selector witness) := by
  exact .transfer (LeftBound69729.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound69729.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound69729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound69729.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69730

namespace LeftBound69809
def owner : Owner := ⟨.program ⟨214⟩, ⟨14416⟩⟩
def transferEvent : Nat := 69809
def frameStart : Nat := 69780
def rule : BoundRule := .product (.predecessor 0 69807 .coefficient) (.predecessor 1 69808 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69807 .coefficient)
      LeftAuthority69805.bound (LeftAuthority69805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69805.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69805.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69808 .coefficient)
      LeftAuthority69802.bound (LeftAuthority69802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69802.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority69805.bound LeftAuthority69802.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69805.bound, LeftAuthority69802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority69805.actual selector witness) * (LeftAuthority69802.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69809

namespace LeftBound69813
def owner : Owner := ⟨.program ⟨214⟩, ⟨14417⟩⟩
def transferEvent : Nat := 69813
def frameStart : Nat := 69780
def rule : BoundRule := .identity (.predecessor 0 69812 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69812 .coefficient)
      LeftBound69809.bound (LeftBound69809.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69811RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69809.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69809.derived selector witness)

def rawBound : CoeffClass := LeftBound69809.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69809.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound69809.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69813

namespace LeftBound69830
def owner : Owner := ⟨.program ⟨214⟩, ⟨14527⟩⟩
def transferEvent : Nat := 69830
def frameStart : Nat := 69780
def rule : BoundRule := .sum [.predecessor 0 69828 .coefficient, .predecessor 1 69829 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69828 .coefficient)
      LeftBound69813.bound (LeftBound69813.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69813.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69829 .coefficient)
      LeftAuthority69826.bound (LeftAuthority69826.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority69826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69813.bound, LeftAuthority69826.bound]
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69813.bound, LeftAuthority69826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69813.actual selector witness, LeftAuthority69826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69830

namespace LeftBound69833
def owner : Owner := ⟨.program ⟨214⟩, ⟨14528⟩⟩
def transferEvent : Nat := 69833
def frameStart : Nat := 69780
def rule : BoundRule := .identity (.predecessor 0 69832 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69832 .coefficient)
      LeftBound69830.bound (LeftBound69830.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound69830.derived selector witness)

def rawBound : CoeffClass := LeftBound69830.bound
def bound : CoeffClass := .finite ⟨484, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69830.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound69830.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69833

namespace LeftBound69839
def owner : Owner := ⟨.program ⟨214⟩, ⟨14529⟩⟩
def transferEvent : Nat := 69839
def frameStart : Nat := 69780
def rule : BoundRule := .product (.predecessor 0 69837 .coefficient) (.predecessor 1 69838 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69837 .coefficient)
      LeftAuthority69835.bound (LeftAuthority69835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69835.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69838 .coefficient)
      LeftBound69833.bound (LeftBound69833.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69834RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69833.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69833.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority69835.bound LeftBound69833.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69835.bound, LeftBound69833.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority69835.actual selector witness) * (LeftBound69833.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69839

namespace LeftBound69855
def owner : Owner := ⟨.program ⟨214⟩, ⟨7856⟩⟩
def transferEvent : Nat := 69855
def frameStart : Nat := 69780
def rule : BoundRule := .scale (.predecessor 0 69853 .coefficient) (.value (.predecessor 1 69854 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69853 .coefficient)
      LeftAuthority69851.bound (LeftAuthority69851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69851.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69854 .coefficient)
      LeftAuthority69842.bound (LeftAuthority69842.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority69842.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority69851.bound LeftAuthority69842.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69851.bound, LeftAuthority69842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority69851.actual selector witness) * (LeftAuthority69842.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound69855

namespace LeftBound69858
def owner : Owner := ⟨.program ⟨214⟩, ⟨6761⟩⟩
def transferEvent : Nat := 69858
def frameStart : Nat := 69780
def rule : BoundRule := .identity (.predecessor 0 69857 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69857 .coefficient)
      LeftAuthority69845.bound (LeftAuthority69845.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69846RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority69845.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority69845.derived selector witness)

def rawBound : CoeffClass := LeftAuthority69845.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority69845.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority69845.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound69858

namespace LeftBound69862
def owner : Owner := ⟨.program ⟨214⟩, ⟨7857⟩⟩
def transferEvent : Nat := 69862
def frameStart : Nat := 69780
def rule : BoundRule := .product (.predecessor 0 69860 .coefficient) (.predecessor 1 69861 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69860 .coefficient)
      LeftBound69858.bound (LeftBound69858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69858.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69861 .coefficient)
      LeftBound69855.bound (LeftBound69855.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69856RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69855.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69855.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound69858.bound LeftBound69855.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69858.bound, LeftBound69855.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound69858.actual selector witness) * (LeftBound69855.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound69862

namespace LeftBound69867
def owner : Owner := ⟨.program ⟨214⟩, ⟨14530⟩⟩
def transferEvent : Nat := 69867
def frameStart : Nat := 69780
def rule : BoundRule := .sum [.predecessor 0 69865 .coefficient, .predecessor 1 69866 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 69865 .coefficient)
      LeftBound69862.bound (LeftBound69862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69864RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69862.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 69866 .coefficient)
      LeftBound69839.bound (LeftBound69839.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events272.exact69841RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound69839.bound, RecordedBoundRefines] <;> decide)
      (LeftBound69839.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound69862.bound, LeftBound69839.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound69862.bound, LeftBound69839.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound69862.actual selector witness, LeftBound69839.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound69867

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
