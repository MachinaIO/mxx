import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard589

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound86239
def owner : Owner := ⟨.program ⟨214⟩, ⟨25836⟩⟩
def transferEvent : Nat := 86239
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨25835⟩⟩]⟩ [⟨.result 86171 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86171 .coefficient)
      LeftAuthority86170.bound (LeftAuthority86170.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨25835⟩⟩) (rawTerms := some (Proof.Events336.exact86171RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86170.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86170.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86170.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86239

namespace LeftBound86240
def owner : Owner := ⟨.program ⟨214⟩, ⟨25836⟩⟩
def transferEvent : Nat := 86240
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 86235 .summary) (.transfer 86239) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86235 .summary)
      LeftBound86234.bound (LeftBound86234.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨13564⟩⟩) (rawTerms := some (Proof.Events336.exact86235RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound86234.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86239)
      LeftBound86239.bound (LeftBound86239.actual selector witness) := by
  exact .transfer (LeftBound86239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86234.bound LeftBound86239.bound
def bound : CoeffClass := .finite ⟨350224987979776, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86234.bound, LeftBound86239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86234.actual selector witness) * (LeftBound86239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86240

namespace LeftBound86251
def owner : Owner := ⟨.program ⟨214⟩, ⟨19314⟩⟩
def transferEvent : Nat := 86251
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 86249 .coefficient) (.value (.predecessor 1 86250 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86249 .coefficient)
      LeftAuthority86247.bound (LeftAuthority86247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86250 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority86247.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86247.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86247.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86251

namespace LeftBound86255
def owner : Owner := ⟨.program ⟨214⟩, ⟨19315⟩⟩
def transferEvent : Nat := 86255
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 86253 .coefficient) (.predecessor 1 86254 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86253 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86254 .coefficient)
      LeftBound86251.bound (LeftBound86251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86251.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound86251.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound86251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound86251.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86255

namespace LeftBound86256
def owner : Owner := ⟨.program ⟨214⟩, ⟨19315⟩⟩
def transferEvent : Nat := 86256
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨19312⟩⟩]⟩ [⟨.result 86248 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 86248 .coefficient)
      LeftAuthority86247.bound (LeftAuthority86247.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨19312⟩⟩) (rawTerms := some (Proof.Events336.exact86248RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86247.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority86247.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86247.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound86256

namespace LeftBound86257
def owner : Owner := ⟨.program ⟨214⟩, ⟨19315⟩⟩
def transferEvent : Nat := 86257
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 86256) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 86256)
      LeftBound86256.bound (LeftBound86256.actual selector witness) := by
  exact .transfer (LeftBound86256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound86256.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound86256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound86256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86257

namespace LeftBound86336
def owner : Owner := ⟨.program ⟨214⟩, ⟨13557⟩⟩
def transferEvent : Nat := 86336
def frameStart : Nat := 86307
def rule : BoundRule := .product (.predecessor 0 86334 .coefficient) (.predecessor 1 86335 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86334 .coefficient)
      LeftAuthority86332.bound (LeftAuthority86332.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86333RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86332.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86332.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86335 .coefficient)
      LeftAuthority86329.bound (LeftAuthority86329.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86330RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86329.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86329.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority86332.bound LeftAuthority86329.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86332.bound, LeftAuthority86329.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority86332.actual selector witness) * (LeftAuthority86329.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86336

namespace LeftBound86340
def owner : Owner := ⟨.program ⟨214⟩, ⟨13558⟩⟩
def transferEvent : Nat := 86340
def frameStart : Nat := 86307
def rule : BoundRule := .identity (.predecessor 0 86339 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86339 .coefficient)
      LeftBound86336.bound (LeftBound86336.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86338RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86336.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86336.derived selector witness)

def rawBound : CoeffClass := LeftBound86336.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86336.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86340

namespace LeftBound86357
def owner : Owner := ⟨.program ⟨214⟩, ⟨13663⟩⟩
def transferEvent : Nat := 86357
def frameStart : Nat := 86307
def rule : BoundRule := .sum [.predecessor 0 86355 .coefficient, .predecessor 1 86356 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86355 .coefficient)
      LeftBound86340.bound (LeftBound86340.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86356 .coefficient)
      LeftAuthority86353.bound (LeftAuthority86353.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86353.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86340.bound, LeftAuthority86353.bound]
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86340.bound, LeftAuthority86353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86340.actual selector witness, LeftAuthority86353.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86357

namespace LeftBound86360
def owner : Owner := ⟨.program ⟨214⟩, ⟨13664⟩⟩
def transferEvent : Nat := 86360
def frameStart : Nat := 86307
def rule : BoundRule := .identity (.predecessor 0 86359 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86359 .coefficient)
      LeftBound86357.bound (LeftBound86357.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound86357.derived selector witness)

def rawBound : CoeffClass := LeftBound86357.bound
def bound : CoeffClass := .finite ⟨100, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86357.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound86357.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86360

namespace LeftBound86366
def owner : Owner := ⟨.program ⟨214⟩, ⟨13665⟩⟩
def transferEvent : Nat := 86366
def frameStart : Nat := 86307
def rule : BoundRule := .product (.predecessor 0 86364 .coefficient) (.predecessor 1 86365 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86364 .coefficient)
      LeftAuthority86362.bound (LeftAuthority86362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86362.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86365 .coefficient)
      LeftBound86360.bound (LeftBound86360.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86361RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86360.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86360.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority86362.bound LeftBound86360.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86362.bound, LeftBound86360.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority86362.actual selector witness) * (LeftBound86360.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86366

namespace LeftBound86380
def owner : Owner := ⟨.program ⟨214⟩, ⟨7844⟩⟩
def transferEvent : Nat := 86380
def frameStart : Nat := 86307
def rule : BoundRule := .scale (.predecessor 0 86378 .coefficient) (.value (.predecessor 1 86379 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86378 .coefficient)
      LeftAuthority86376.bound (LeftAuthority86376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86376.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86379 .coefficient)
      LeftAuthority86310.bound (LeftAuthority86310.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority86310.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority86376.bound LeftAuthority86310.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86376.bound, LeftAuthority86310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority86376.actual selector witness) * (LeftAuthority86310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound86380

namespace LeftBound86383
def owner : Owner := ⟨.program ⟨214⟩, ⟨6793⟩⟩
def transferEvent : Nat := 86383
def frameStart : Nat := 86307
def rule : BoundRule := .identity (.predecessor 0 86382 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86382 .coefficient)
      LeftAuthority86370.bound (LeftAuthority86370.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86371RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86370.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86370.derived selector witness)

def rawBound : CoeffClass := LeftAuthority86370.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority86370.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound86383

namespace LeftBound86387
def owner : Owner := ⟨.program ⟨214⟩, ⟨7845⟩⟩
def transferEvent : Nat := 86387
def frameStart : Nat := 86307
def rule : BoundRule := .product (.predecessor 0 86385 .coefficient) (.predecessor 1 86386 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86385 .coefficient)
      LeftBound86383.bound (LeftBound86383.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86384RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86383.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86383.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86386 .coefficient)
      LeftBound86380.bound (LeftBound86380.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86381RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86380.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86380.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86383.bound LeftBound86380.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86383.bound, LeftBound86380.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86383.actual selector witness) * (LeftBound86380.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86387

namespace LeftBound86392
def owner : Owner := ⟨.program ⟨214⟩, ⟨13666⟩⟩
def transferEvent : Nat := 86392
def frameStart : Nat := 86307
def rule : BoundRule := .sum [.predecessor 0 86390 .coefficient, .predecessor 1 86391 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86390 .coefficient)
      LeftBound86387.bound (LeftBound86387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86389RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86387.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86391 .coefficient)
      LeftBound86366.bound (LeftBound86366.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86366.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86366.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86387.bound, LeftBound86366.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86387.bound, LeftBound86366.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound86387.actual selector witness, LeftBound86366.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86392

namespace LeftBound86396
def owner : Owner := ⟨.program ⟨214⟩, ⟨25838⟩⟩
def transferEvent : Nat := 86396
def frameStart : Nat := 86307
def rule : BoundRule := .product (.predecessor 0 86394 .coefficient) (.predecessor 1 86395 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 86394 .coefficient)
      LeftBound86392.bound (LeftBound86392.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86392.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86392.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 86395 .coefficient)
      LeftAuthority86351.bound (LeftAuthority86351.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events337.exact86352RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86351.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86351.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound86392.bound LeftAuthority86351.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86392.bound, LeftAuthority86351.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound86392.actual selector witness) * (LeftAuthority86351.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86396

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
