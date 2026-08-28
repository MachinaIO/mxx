import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard272

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound48260
def owner : Owner := ⟨.program ⟨214⟩, ⟨28104⟩⟩
def transferEvent : Nat := 48260
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48258 .coefficient) (.predecessor 1 48259 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48258 .coefficient)
      LeftBound40657.bound (LeftBound40657.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events158.exact40661RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound40657.bound, RecordedBoundRefines] <;> decide)
      (LeftBound40657.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48259 .coefficient)
      LeftAuthority48256.bound (LeftAuthority48256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40657.bound LeftAuthority48256.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40657.bound, LeftAuthority48256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40657.actual selector witness) * (LeftAuthority48256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48260

namespace LeftBound48261
def owner : Owner := ⟨.program ⟨214⟩, ⟨28104⟩⟩
def transferEvent : Nat := 48261
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28102⟩⟩]⟩ [⟨.result 48257 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48257 .coefficient)
      LeftAuthority48256.bound (LeftAuthority48256.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28102⟩⟩) (rawTerms := some (Proof.Events188.exact48257RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48256.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48256.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48256.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48256.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48261

namespace LeftBound48262
def owner : Owner := ⟨.program ⟨214⟩, ⟨28104⟩⟩
def transferEvent : Nat := 48262
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 40661 .summary) (.transfer 48261) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 40661 .summary)
      LeftBound40660.bound (LeftBound40660.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26155⟩⟩) (rawTerms := some (Proof.Events158.exact40661RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound40660.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48261)
      LeftBound48261.bound (LeftBound48261.actual selector witness) := by
  exact .transfer (LeftBound48261.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound40660.bound LeftBound48261.bound
def bound : CoeffClass := .finite ⟨1292113297018323992576, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound40660.bound, LeftBound48261.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound40660.actual selector witness) * (LeftBound48261.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48262

namespace LeftBound48273
def owner : Owner := ⟨.program ⟨214⟩, ⟨21482⟩⟩
def transferEvent : Nat := 48273
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 48271 .coefficient) (.value (.predecessor 1 48272 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48271 .coefficient)
      LeftAuthority48269.bound (LeftAuthority48269.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48269.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48272 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority48269.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48269.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48269.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound48273

namespace LeftBound48277
def owner : Owner := ⟨.program ⟨214⟩, ⟨21483⟩⟩
def transferEvent : Nat := 48277
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 48275 .coefficient) (.predecessor 1 48276 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48275 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48276 .coefficient)
      LeftBound48273.bound (LeftBound48273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48274RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48273.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound48273.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound48273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound48273.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48277

namespace LeftBound48278
def owner : Owner := ⟨.program ⟨214⟩, ⟨21483⟩⟩
def transferEvent : Nat := 48278
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨21480⟩⟩]⟩ [⟨.result 48270 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 48270 .coefficient)
      LeftAuthority48269.bound (LeftAuthority48269.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨21480⟩⟩) (rawTerms := some (Proof.Events188.exact48270RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48269.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48269.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority48269.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority48269.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound48278

namespace LeftBound48279
def owner : Owner := ⟨.program ⟨214⟩, ⟨21483⟩⟩
def transferEvent : Nat := 48279
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 48278) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 48278)
      LeftBound48278.bound (LeftBound48278.actual selector witness) := by
  exact .transfer (LeftBound48278.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound48278.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound48278.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound48278.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48279

namespace LeftBound48374
def owner : Owner := ⟨.program ⟨214⟩, ⟨16068⟩⟩
def transferEvent : Nat := 48374
def frameStart : Nat := 48335
def rule : BoundRule := .identity (.predecessor 0 48373 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48373 .coefficient)
      LeftAuthority48371.bound (LeftAuthority48371.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events188.exact48372RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48371.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48371.derived selector witness)

def rawBound : CoeffClass := LeftAuthority48371.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48371.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority48371.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48374

namespace LeftBound48391
def owner : Owner := ⟨.program ⟨214⟩, ⟨16142⟩⟩
def transferEvent : Nat := 48391
def frameStart : Nat := 48335
def rule : BoundRule := .sum [.predecessor 0 48389 .coefficient, .predecessor 1 48390 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48389 .coefficient)
      LeftBound48374.bound (LeftBound48374.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48390 .coefficient)
      LeftAuthority48387.bound (LeftAuthority48387.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority48387.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48374.bound, LeftAuthority48387.bound]
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48374.bound, LeftAuthority48387.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48374.actual selector witness, LeftAuthority48387.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48391

namespace LeftBound48394
def owner : Owner := ⟨.program ⟨214⟩, ⟨16143⟩⟩
def transferEvent : Nat := 48394
def frameStart : Nat := 48335
def rule : BoundRule := .identity (.predecessor 0 48393 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48393 .coefficient)
      LeftBound48391.bound (LeftBound48391.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound48391.derived selector witness)

def rawBound : CoeffClass := LeftBound48391.bound
def bound : CoeffClass := .finite ⟨22, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound48391.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound48394

namespace LeftBound48400
def owner : Owner := ⟨.program ⟨214⟩, ⟨16144⟩⟩
def transferEvent : Nat := 48400
def frameStart : Nat := 48335
def rule : BoundRule := .product (.predecessor 0 48398 .coefficient) (.predecessor 1 48399 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48398 .coefficient)
      LeftAuthority48396.bound (LeftAuthority48396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48399 .coefficient)
      LeftBound48394.bound (LeftBound48394.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48394.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48394.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority48396.bound LeftBound48394.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48396.bound, LeftBound48394.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority48396.actual selector witness) * (LeftBound48394.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48400

namespace LeftBound48408
def owner : Owner := ⟨.program ⟨214⟩, ⟨16145⟩⟩
def transferEvent : Nat := 48408
def frameStart : Nat := 48335
def rule : BoundRule := .sum [.predecessor 0 48406 .coefficient, .predecessor 1 48407 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48406 .coefficient)
      LeftAuthority48404.bound (LeftAuthority48404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48404.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48407 .coefficient)
      LeftBound48400.bound (LeftBound48400.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48400.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48400.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48404.bound, LeftBound48400.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48404.bound, LeftBound48400.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48404.actual selector witness, LeftBound48400.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48408

namespace LeftBound48412
def owner : Owner := ⟨.program ⟨214⟩, ⟨28103⟩⟩
def transferEvent : Nat := 48412
def frameStart : Nat := 48335
def rule : BoundRule := .product (.predecessor 0 48410 .coefficient) (.predecessor 1 48411 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48410 .coefficient)
      LeftBound48408.bound (LeftBound48408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48408.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48408.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48411 .coefficient)
      LeftAuthority48385.bound (LeftAuthority48385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48385.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48385.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound48408.bound LeftAuthority48385.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48408.bound, LeftAuthority48385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound48408.actual selector witness) * (LeftAuthority48385.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48412

namespace LeftBound48423
def owner : Owner := ⟨.program ⟨214⟩, ⟨18054⟩⟩
def transferEvent : Nat := 48423
def frameStart : Nat := 48335
def rule : BoundRule := .product (.predecessor 0 48421 .coefficient) (.predecessor 1 48422 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48421 .coefficient)
      LeftAuthority48396.bound (LeftAuthority48396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48396.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48396.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48422 .coefficient)
      LeftAuthority48419.bound (LeftAuthority48419.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48420RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48419.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48419.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority48396.bound LeftAuthority48419.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48396.bound, LeftAuthority48419.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority48396.actual selector witness) * (LeftAuthority48419.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound48423

namespace LeftBound48431
def owner : Owner := ⟨.program ⟨214⟩, ⟨18055⟩⟩
def transferEvent : Nat := 48431
def frameStart : Nat := 48335
def rule : BoundRule := .sum [.predecessor 0 48429 .coefficient, .predecessor 1 48430 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48429 .coefficient)
      LeftAuthority48427.bound (LeftAuthority48427.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48428RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority48427.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority48427.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48430 .coefficient)
      LeftBound48423.bound (LeftBound48423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority48427.bound, LeftBound48423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority48427.bound, LeftBound48423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority48427.actual selector witness, LeftBound48423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48431

namespace LeftBound48435
def owner : Owner := ⟨.program ⟨214⟩, ⟨28108⟩⟩
def transferEvent : Nat := 48435
def frameStart : Nat := 48335
def rule : BoundRule := .sum [.predecessor 0 48433 .coefficient, .predecessor 1 48434 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 48433 .coefficient)
      LeftBound48431.bound (LeftBound48431.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48431.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48431.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 48434 .coefficient)
      LeftBound48412.bound (LeftBound48412.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events189.exact48417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound48412.bound, RecordedBoundRefines] <;> decide)
      (LeftBound48412.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound48431.bound, LeftBound48412.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound48431.bound, LeftBound48412.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound48431.actual selector witness, LeftBound48412.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound48435

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
