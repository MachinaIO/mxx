import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard002
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard025

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound6326
def owner : Owner := ⟨.program ⟨214⟩, ⟨7768⟩⟩
def transferEvent : Nat := 6326
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 6324 .coefficient, .predecessor 1 6325 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6324 .coefficient)
      LeftBound6322.bound (LeftBound6322.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6322.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6325 .coefficient)
      LeftAuthority6276.bound (LeftAuthority6276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6276.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6322.bound, LeftAuthority6276.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6322.bound, LeftAuthority6276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6322.actual selector witness, LeftAuthority6276.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6326

namespace LeftBound6327
def owner : Owner := ⟨.program ⟨214⟩, ⟨7768⟩⟩
def transferEvent : Nat := 6327
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨5⟩⟩]⟩ [⟨.result 6277 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 6277 .coefficient)
      LeftAuthority6276.bound (LeftAuthority6276.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨5⟩⟩) (rawTerms := some (Proof.Events024.exact6277RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6276.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6276.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority6276.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority6276.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound6327

namespace LeftBound6332
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6332
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 6330 .coefficient) (.predecessor 1 6331 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 6330 .coefficient)
      LeftBound6326.bound (LeftBound6326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events024.exact6329RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound6326.bound, RecordedBoundRefines] <;> decide)
      (LeftBound6326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 6331 .coefficient)
      LeftBound803.bound (LeftBound803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events003.exact804RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound803.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftBound6326.bound LeftBound803.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6326.bound, LeftBound803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftBound6326.actual selector witness) * (LeftBound803.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound6332

namespace LeftBound6333
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6333
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6410⟩⟩, ⟨.program ⟨214⟩, ⟨18511⟩⟩], []⟩ [⟨.result 36 .coefficient, true, some 1⟩, ⟨.result 536 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6410⟩⟩) (rawTerms := some (Proof.Events000.exact36RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 536 .coefficient)
      LeftAuthority535.bound (LeftAuthority535.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18511⟩⟩) (rawTerms := some (Proof.Events002.exact536RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority535.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority535.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35.bound [LeftAuthority535.bound]
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35.bound, LeftAuthority535.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35.actual selector witness) * ([LeftAuthority535.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6333

namespace LeftBound6334
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6334
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6435⟩⟩, ⟨.program ⟨214⟩, ⟨18140⟩⟩], []⟩ [⟨.result 543 .coefficient, true, some 1⟩, ⟨.result 546 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 543 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6435⟩⟩) (rawTerms := some (Proof.Events002.exact543RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 546 .coefficient)
      LeftAuthority545.bound (LeftAuthority545.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨18140⟩⟩) (rawTerms := some (Proof.Events002.exact546RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority545.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority545.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority542.bound [LeftAuthority545.bound]
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority542.bound, LeftAuthority545.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority542.actual selector witness) * ([LeftAuthority545.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6334

namespace LeftBound6335
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6335
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6333, .transfer 6334]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6333)
      LeftBound6333.bound (LeftBound6333.actual selector witness) := by
  exact .transfer (LeftBound6333.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6334)
      LeftBound6334.bound (LeftBound6334.actual selector witness) := by
  exact .transfer (LeftBound6334.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6333.bound, LeftBound6334.bound]
def bound : CoeffClass := .finite ⟨4453112970957156472086120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6333.bound, LeftBound6334.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6333.actual selector witness, LeftBound6334.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6335

namespace LeftBound6336
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6336
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6437⟩⟩, ⟨.program ⟨214⟩, ⟨16943⟩⟩], []⟩ [⟨.result 553 .coefficient, true, some 1⟩, ⟨.result 556 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 553 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6437⟩⟩) (rawTerms := some (Proof.Events002.exact553RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 556 .coefficient)
      LeftAuthority555.bound (LeftAuthority555.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨16943⟩⟩) (rawTerms := some (Proof.Events002.exact556RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority555.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority555.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority552.bound [LeftAuthority555.bound]
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority552.bound, LeftAuthority555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority552.actual selector witness) * ([LeftAuthority555.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6336

namespace LeftBound6337
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6337
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6335, .transfer 6336]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6335)
      LeftBound6335.bound (LeftBound6335.actual selector witness) := by
  exact .transfer (LeftBound6335.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6336)
      LeftBound6336.bound (LeftBound6336.actual selector witness) := by
  exact .transfer (LeftBound6336.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6335.bound, LeftBound6336.bound]
def bound : CoeffClass := .finite ⟨4683713856341753228595600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6335.bound, LeftBound6336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6335.actual selector witness, LeftBound6336.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6337

namespace LeftBound6338
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6338
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6449⟩⟩, ⟨.program ⟨214⟩, ⟨17510⟩⟩], []⟩ [⟨.result 563 .coefficient, true, some 1⟩, ⟨.result 566 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 563 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6449⟩⟩) (rawTerms := some (Proof.Events002.exact563RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 566 .coefficient)
      LeftAuthority565.bound (LeftAuthority565.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17510⟩⟩) (rawTerms := some (Proof.Events002.exact566RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority565.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority562.bound [LeftAuthority565.bound]
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority562.bound, LeftAuthority565.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority562.actual selector witness) * ([LeftAuthority565.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6338

namespace LeftBound6339
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6339
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6337, .transfer 6338]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6337)
      LeftBound6337.bound (LeftBound6337.actual selector witness) := by
  exact .transfer (LeftBound6337.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6338)
      LeftBound6338.bound (LeftBound6338.actual selector witness) := by
  exact .transfer (LeftBound6338.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6337.bound, LeftBound6338.bound]
def bound : CoeffClass := .finite ⟨4913864642405495209392960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6337.bound, LeftBound6338.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6337.actual selector witness, LeftBound6338.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6339

namespace LeftBound6340
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6340
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6459⟩⟩, ⟨.program ⟨214⟩, ⟨17734⟩⟩], []⟩ [⟨.result 573 .coefficient, true, some 1⟩, ⟨.result 576 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 573 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6459⟩⟩) (rawTerms := some (Proof.Events002.exact573RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 576 .coefficient)
      LeftAuthority575.bound (LeftAuthority575.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17734⟩⟩) (rawTerms := some (Proof.Events002.exact576RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority575.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority575.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority572.bound [LeftAuthority575.bound]
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority572.bound, LeftAuthority575.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority572.actual selector witness) * ([LeftAuthority575.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6340

namespace LeftBound6341
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6341
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6339, .transfer 6340]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6339)
      LeftBound6339.bound (LeftBound6339.actual selector witness) := by
  exact .transfer (LeftBound6339.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6340)
      LeftBound6340.bound (LeftBound6340.actual selector witness) := by
  exact .transfer (LeftBound6340.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6339.bound, LeftBound6340.bound]
def bound : CoeffClass := .finite ⟨5143450410172845024934680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6339.bound, LeftBound6340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6339.actual selector witness, LeftBound6340.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6341

namespace LeftBound6342
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6342
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6467⟩⟩, ⟨.program ⟨214⟩, ⟨17965⟩⟩], []⟩ [⟨.result 583 .coefficient, true, some 1⟩, ⟨.result 586 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 583 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6467⟩⟩) (rawTerms := some (Proof.Events002.exact583RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 586 .coefficient)
      LeftAuthority585.bound (LeftAuthority585.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17965⟩⟩) (rawTerms := some (Proof.Events002.exact586RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority585.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority582.bound [LeftAuthority585.bound]
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority582.bound, LeftAuthority585.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority582.actual selector witness) * ([LeftAuthority585.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6342

namespace LeftBound6343
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6343
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6341, .transfer 6342]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6341)
      LeftBound6341.bound (LeftBound6341.actual selector witness) := by
  exact .transfer (LeftBound6341.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6342)
      LeftBound6342.bound (LeftBound6342.actual selector witness) := by
  exact .transfer (LeftBound6342.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6341.bound, LeftBound6342.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6341.bound, LeftBound6342.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6341.actual selector witness, LeftBound6342.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6343

namespace LeftBound6344
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6344
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨214⟩, ⟨6473⟩⟩, ⟨.program ⟨214⟩, ⟨17566⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 596 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6473⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 596 .coefficient)
      LeftAuthority595.bound (LeftAuthority595.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨17566⟩⟩) (rawTerms := some (Proof.Events002.exact596RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority595.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority595.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority595.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound6344

namespace LeftBound6345
def owner : Owner := ⟨.program ⟨214⟩, ⟨18906⟩⟩
def transferEvent : Nat := 6345
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 6343, .transfer 6344]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6343)
      LeftBound6343.bound (LeftBound6343.actual selector witness) := by
  exact .transfer (LeftBound6343.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 6344)
      LeftBound6344.bound (LeftBound6344.actual selector witness) := by
  exact .transfer (LeftBound6344.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound6343.bound, LeftBound6344.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound6343.bound, LeftBound6344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound6343.actual selector witness, LeftBound6344.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound6345

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
