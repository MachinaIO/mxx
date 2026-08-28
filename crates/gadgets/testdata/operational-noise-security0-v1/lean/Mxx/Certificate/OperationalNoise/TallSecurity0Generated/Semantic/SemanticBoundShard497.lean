import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard495
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard496

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound72799
def owner : Owner := ⟨.program ⟨214⟩, ⟨25062⟩⟩
def transferEvent : Nat := 72799
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 72797 .coefficient, .predecessor 1 72798 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72797 .coefficient)
      LeftBound72620.bound (LeftBound72620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72796RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72620.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72798 .coefficient)
      LeftBound72603.bound (LeftBound72603.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72603.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72620.bound, LeftBound72603.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72620.bound, LeftBound72603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72620.actual selector witness, LeftBound72603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72799

namespace LeftBound72802
def owner : Owner := ⟨.program ⟨214⟩, ⟨25062⟩⟩
def transferEvent : Nat := 72802
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 72796 .summary, .result 72610 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72796 .summary)
      LeftBound72622.bound (LeftBound72622.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨19167⟩⟩) (rawTerms := some (Proof.Events284.exact72796RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72610 .summary)
      LeftBound72605.bound (LeftBound72605.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25061⟩⟩) (rawTerms := some (Proof.Events283.exact72610RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72622.bound, LeftBound72605.bound]
def bound : CoeffClass := .finite ⟨352017970769920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72622.bound, LeftBound72605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72622.actual selector witness, LeftBound72605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72802

namespace LeftBound72806
def owner : Owner := ⟨.program ⟨214⟩, ⟨26770⟩⟩
def transferEvent : Nat := 72806
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72804 .coefficient) (.predecessor 1 72805 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72804 .coefficient)
      LeftBound72799.bound (LeftBound72799.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72799.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72799.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72805 .coefficient)
      LeftAuthority72525.bound (LeftAuthority72525.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events283.exact72526RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72525.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72799.bound LeftAuthority72525.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72799.bound, LeftAuthority72525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72799.actual selector witness) * (LeftAuthority72525.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72806

namespace LeftBound72807
def owner : Owner := ⟨.program ⟨214⟩, ⟨26770⟩⟩
def transferEvent : Nat := 72807
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26768⟩⟩]⟩ [⟨.result 72526 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72526 .coefficient)
      LeftAuthority72525.bound (LeftAuthority72525.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26768⟩⟩) (rawTerms := some (Proof.Events283.exact72526RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72525.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72525.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72525.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72525.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72525.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72807

namespace LeftBound72808
def owner : Owner := ⟨.program ⟨214⟩, ⟨26770⟩⟩
def transferEvent : Nat := 72808
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 72803 .summary) (.transfer 72807) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72803 .summary)
      LeftBound72802.bound (LeftBound72802.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25062⟩⟩) (rawTerms := some (Proof.Events284.exact72803RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound72802.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72807)
      LeftBound72807.bound (LeftBound72807.actual selector witness) := by
  exact .transfer (LeftBound72807.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72802.bound LeftBound72807.bound
def bound : CoeffClass := .finite ⟨1291911585013138718720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72802.bound, LeftBound72807.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72802.actual selector witness) * (LeftBound72807.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72808

namespace LeftBound72819
def owner : Owner := ⟨.program ⟨214⟩, ⟨20678⟩⟩
def transferEvent : Nat := 72819
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 72817 .coefficient) (.value (.predecessor 1 72818 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72817 .coefficient)
      LeftAuthority72815.bound (LeftAuthority72815.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72816RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72815.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72818 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority72815.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72815.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72815.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound72819

namespace LeftBound72823
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def transferEvent : Nat := 72823
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 72821 .coefficient) (.predecessor 1 72822 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72821 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72822 .coefficient)
      LeftBound72819.bound (LeftBound72819.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72820RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72819.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72819.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound72819.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound72819.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound72819.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72823

namespace LeftBound72824
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def transferEvent : Nat := 72824
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20676⟩⟩]⟩ [⟨.result 72816 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 72816 .coefficient)
      LeftAuthority72815.bound (LeftAuthority72815.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20676⟩⟩) (rawTerms := some (Proof.Events284.exact72816RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72815.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72815.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority72815.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72815.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority72815.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound72824

namespace LeftBound72825
def owner : Owner := ⟨.program ⟨214⟩, ⟨20679⟩⟩
def transferEvent : Nat := 72825
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 72824) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 72824)
      LeftBound72824.bound (LeftBound72824.actual selector witness) := by
  exact .transfer (LeftBound72824.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound72824.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound72824.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound72824.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72825

namespace LeftBound72920
def owner : Owner := ⟨.program ⟨214⟩, ⟨15111⟩⟩
def transferEvent : Nat := 72920
def frameStart : Nat := 72881
def rule : BoundRule := .identity (.predecessor 0 72919 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72919 .coefficient)
      LeftAuthority72917.bound (LeftAuthority72917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72917.derived selector witness)

def rawBound : CoeffClass := LeftAuthority72917.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority72917.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72920

namespace LeftBound72937
def owner : Owner := ⟨.program ⟨214⟩, ⟨15150⟩⟩
def transferEvent : Nat := 72937
def frameStart : Nat := 72881
def rule : BoundRule := .sum [.predecessor 0 72935 .coefficient, .predecessor 1 72936 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72935 .coefficient)
      LeftBound72920.bound (LeftBound72920.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72920.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72936 .coefficient)
      LeftAuthority72933.bound (LeftAuthority72933.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority72933.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound72920.bound, LeftAuthority72933.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72920.bound, LeftAuthority72933.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound72920.actual selector witness, LeftAuthority72933.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72937

namespace LeftBound72940
def owner : Owner := ⟨.program ⟨214⟩, ⟨15151⟩⟩
def transferEvent : Nat := 72940
def frameStart : Nat := 72881
def rule : BoundRule := .identity (.predecessor 0 72939 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72939 .coefficient)
      LeftBound72937.bound (LeftBound72937.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound72937.derived selector witness)

def rawBound : CoeffClass := LeftBound72937.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound72937.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound72940

namespace LeftBound72946
def owner : Owner := ⟨.program ⟨214⟩, ⟨15152⟩⟩
def transferEvent : Nat := 72946
def frameStart : Nat := 72881
def rule : BoundRule := .product (.predecessor 0 72944 .coefficient) (.predecessor 1 72945 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72944 .coefficient)
      LeftAuthority72942.bound (LeftAuthority72942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72945 .coefficient)
      LeftBound72940.bound (LeftBound72940.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72941RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72940.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72940.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority72942.bound LeftBound72940.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72942.bound, LeftBound72940.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority72942.actual selector witness) * (LeftBound72940.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72946

namespace LeftBound72954
def owner : Owner := ⟨.program ⟨214⟩, ⟨15153⟩⟩
def transferEvent : Nat := 72954
def frameStart : Nat := 72881
def rule : BoundRule := .sum [.predecessor 0 72952 .coefficient, .predecessor 1 72953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72952 .coefficient)
      LeftAuthority72950.bound (LeftAuthority72950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72950.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72953 .coefficient)
      LeftBound72946.bound (LeftBound72946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72946.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority72950.bound, LeftBound72946.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72950.bound, LeftBound72946.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority72950.actual selector witness, LeftBound72946.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound72954

namespace LeftBound72958
def owner : Owner := ⟨.program ⟨214⟩, ⟨26769⟩⟩
def transferEvent : Nat := 72958
def frameStart : Nat := 72881
def rule : BoundRule := .product (.predecessor 0 72956 .coefficient) (.predecessor 1 72957 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72956 .coefficient)
      LeftBound72954.bound (LeftBound72954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound72954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound72954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72957 .coefficient)
      LeftAuthority72931.bound (LeftAuthority72931.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72931.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72931.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound72954.bound LeftAuthority72931.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound72954.bound, LeftAuthority72931.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound72954.actual selector witness) * (LeftAuthority72931.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72958

namespace LeftBound72969
def owner : Owner := ⟨.program ⟨214⟩, ⟨15364⟩⟩
def transferEvent : Nat := 72969
def frameStart : Nat := 72881
def rule : BoundRule := .product (.predecessor 0 72967 .coefficient) (.predecessor 1 72968 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 72967 .coefficient)
      LeftAuthority72942.bound (LeftAuthority72942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events284.exact72943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72942.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 72968 .coefficient)
      LeftAuthority72965.bound (LeftAuthority72965.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events285.exact72966RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority72965.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority72965.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority72942.bound LeftAuthority72965.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority72942.bound, LeftAuthority72965.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority72942.actual selector witness) * (LeftAuthority72965.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound72969

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
