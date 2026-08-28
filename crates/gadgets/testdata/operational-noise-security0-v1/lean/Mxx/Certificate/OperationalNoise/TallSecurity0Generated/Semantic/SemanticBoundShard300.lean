import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard299

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound44311
def owner : Owner := ⟨.program ⟨214⟩, ⟨10503⟩⟩
def transferEvent : Nat := 44311
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 44309 .coefficient, .predecessor 1 44310 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44309 .coefficient)
      LeftBound44301.bound (LeftBound44301.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44308RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44301.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44301.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44310 .coefficient)
      LeftBound44273.bound (LeftBound44273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44273.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44301.bound, LeftBound44273.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44301.bound, LeftBound44273.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44301.actual selector witness, LeftBound44273.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44311

namespace LeftBound44313
def owner : Owner := ⟨.program ⟨214⟩, ⟨10503⟩⟩
def transferEvent : Nat := 44313
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 44308 .summary, .result 44278 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44308 .summary)
      LeftBound44303.bound (LeftBound44303.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨9414⟩⟩) (rawTerms := some (Proof.Events173.exact44308RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44303.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44278 .summary)
      LeftBound44275.bound (LeftBound44275.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10502⟩⟩) (rawTerms := some (Proof.Events172.exact44278RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44275.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44303.bound, LeftBound44275.bound]
def bound : CoeffClass := .finite ⟨95422080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44303.bound, LeftBound44275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44303.actual selector witness, LeftBound44275.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44313

namespace LeftBound44317
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def transferEvent : Nat := 44317
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44315 .coefficient) (.predecessor 1 44316 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44315 .coefficient)
      LeftBound44311.bound (LeftBound44311.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44314RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44311.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44311.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44316 .coefficient)
      LeftAuthority44249.bound (LeftAuthority44249.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events172.exact44250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44249.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44311.bound LeftAuthority44249.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44311.bound, LeftAuthority44249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44311.actual selector witness) * (LeftAuthority44249.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44317

namespace LeftBound44318
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def transferEvent : Nat := 44318
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨24921⟩⟩]⟩ [⟨.result 44250 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44250 .coefficient)
      LeftAuthority44249.bound (LeftAuthority44249.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨24921⟩⟩) (rawTerms := some (Proof.Events172.exact44250RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44249.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44249.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44249.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44249.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44249.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44318

namespace LeftBound44319
def owner : Owner := ⟨.program ⟨214⟩, ⟨24922⟩⟩
def transferEvent : Nat := 44319
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 44314 .summary) (.transfer 44318) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44314 .summary)
      LeftBound44313.bound (LeftBound44313.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨10503⟩⟩) (rawTerms := some (Proof.Events173.exact44314RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound44313.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44318)
      LeftBound44318.bound (LeftBound44318.actual selector witness) := by
  exact .transfer (LeftBound44318.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound44313.bound LeftBound44318.bound
def bound : CoeffClass := .finite ⟨350200560353280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44313.bound, LeftBound44318.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound44313.actual selector witness) * (LeftBound44318.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44319

namespace LeftBound44330
def owner : Owner := ⟨.program ⟨214⟩, ⟨19034⟩⟩
def transferEvent : Nat := 44330
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 44328 .coefficient) (.value (.predecessor 1 44329 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44328 .coefficient)
      LeftAuthority44326.bound (LeftAuthority44326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44329 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44326.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44326.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44326.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44330

namespace LeftBound44334
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def transferEvent : Nat := 44334
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 44332 .coefficient) (.predecessor 1 44333 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44332 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44333 .coefficient)
      LeftBound44330.bound (LeftBound44330.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44331RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44330.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44330.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound44330.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound44330.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound44330.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44334

namespace LeftBound44335
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def transferEvent : Nat := 44335
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19032⟩⟩]⟩ [⟨.result 44327 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 44327 .coefficient)
      LeftAuthority44326.bound (LeftAuthority44326.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19032⟩⟩) (rawTerms := some (Proof.Events173.exact44327RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44326.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority44326.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44326.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44326.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound44335

namespace LeftBound44336
def owner : Owner := ⟨.program ⟨214⟩, ⟨19035⟩⟩
def transferEvent : Nat := 44336
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 44335) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 44335)
      LeftBound44335.bound (LeftBound44335.actual selector witness) := by
  exact .transfer (LeftBound44335.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound44335.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound44335.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound44335.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44336

namespace LeftBound44415
def owner : Owner := ⟨.program ⟨214⟩, ⟨10497⟩⟩
def transferEvent : Nat := 44415
def frameStart : Nat := 44386
def rule : BoundRule := .product (.predecessor 0 44413 .coefficient) (.predecessor 1 44414 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44413 .coefficient)
      LeftAuthority44411.bound (LeftAuthority44411.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44412RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44411.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44414 .coefficient)
      LeftAuthority44408.bound (LeftAuthority44408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44408.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority44411.bound LeftAuthority44408.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44411.bound, LeftAuthority44408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority44411.actual selector witness) * (LeftAuthority44408.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44415

namespace LeftBound44419
def owner : Owner := ⟨.program ⟨214⟩, ⟨10498⟩⟩
def transferEvent : Nat := 44419
def frameStart : Nat := 44386
def rule : BoundRule := .identity (.predecessor 0 44418 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44418 .coefficient)
      LeftBound44415.bound (LeftBound44415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44417RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44415.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44415.derived selector witness)

def rawBound : CoeffClass := LeftBound44415.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound44415.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44419

namespace LeftBound44436
def owner : Owner := ⟨.program ⟨214⟩, ⟨10584⟩⟩
def transferEvent : Nat := 44436
def frameStart : Nat := 44386
def rule : BoundRule := .sum [.predecessor 0 44434 .coefficient, .predecessor 1 44435 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44434 .coefficient)
      LeftBound44419.bound (LeftBound44419.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44419.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44435 .coefficient)
      LeftAuthority44432.bound (LeftAuthority44432.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority44432.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound44419.bound, LeftAuthority44432.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44419.bound, LeftAuthority44432.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound44419.actual selector witness, LeftAuthority44432.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound44436

namespace LeftBound44439
def owner : Owner := ⟨.program ⟨214⟩, ⟨10585⟩⟩
def transferEvent : Nat := 44439
def frameStart : Nat := 44386
def rule : BoundRule := .identity (.predecessor 0 44438 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44438 .coefficient)
      LeftBound44436.bound (LeftBound44436.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound44436.derived selector witness)

def rawBound : CoeffClass := LeftBound44436.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound44436.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound44436.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44439

namespace LeftBound44445
def owner : Owner := ⟨.program ⟨214⟩, ⟨10586⟩⟩
def transferEvent : Nat := 44445
def frameStart : Nat := 44386
def rule : BoundRule := .product (.predecessor 0 44443 .coefficient) (.predecessor 1 44444 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44443 .coefficient)
      LeftAuthority44441.bound (LeftAuthority44441.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44442RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44441.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44441.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44444 .coefficient)
      LeftBound44439.bound (LeftBound44439.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44440RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound44439.bound, RecordedBoundRefines] <;> decide)
      (LeftBound44439.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority44441.bound LeftBound44439.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44441.bound, LeftBound44439.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority44441.actual selector witness) * (LeftBound44439.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound44445

namespace LeftBound44461
def owner : Owner := ⟨.program ⟨214⟩, ⟨7832⟩⟩
def transferEvent : Nat := 44461
def frameStart : Nat := 44386
def rule : BoundRule := .scale (.predecessor 0 44459 .coefficient) (.value (.predecessor 1 44460 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44459 .coefficient)
      LeftAuthority44457.bound (LeftAuthority44457.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44458RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44457.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44457.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 44460 .coefficient)
      LeftAuthority44448.bound (LeftAuthority44448.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority44448.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority44457.bound LeftAuthority44448.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44457.bound, LeftAuthority44448.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority44457.actual selector witness) * (LeftAuthority44448.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound44461

namespace LeftBound44464
def owner : Owner := ⟨.program ⟨214⟩, ⟨6771⟩⟩
def transferEvent : Nat := 44464
def frameStart : Nat := 44386
def rule : BoundRule := .identity (.predecessor 0 44463 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 44463 .coefficient)
      LeftAuthority44451.bound (LeftAuthority44451.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events173.exact44452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority44451.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority44451.derived selector witness)

def rawBound : CoeffClass := LeftAuthority44451.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority44451.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority44451.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound44464

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
