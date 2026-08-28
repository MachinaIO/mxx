import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard542
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard543
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard562
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard618

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound91236
def owner : Owner := ⟨.program ⟨214⟩, ⟨29165⟩⟩
def transferEvent : Nat := 91236
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 91231 .summary) (.transfer 91235) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91231 .summary)
      LeftBound91230.bound (LeftBound91230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨29164⟩⟩) (rawTerms := some (Proof.Events356.exact91231RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound91230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91235)
      LeftBound91235.bound (LeftBound91235.actual selector witness) := by
  exact .transfer (LeftBound91235.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91230.bound LeftBound91235.bound
def bound : CoeffClass := .finite ⟨4742899020835760917459238912, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91230.bound, LeftBound91235.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91230.actual selector witness) * (LeftBound91235.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91236

namespace LeftBound91251
def owner : Owner := ⟨.program ⟨214⟩, ⟨28946⟩⟩
def transferEvent : Nat := 91251
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91249 .coefficient) (.predecessor 1 91250 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91249 .coefficient)
      LeftBound82592.bound (LeftBound82592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events322.exact82596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound82592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound82592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91250 .coefficient)
      LeftAuthority91247.bound (LeftAuthority91247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91247.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82592.bound LeftAuthority91247.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82592.bound, LeftAuthority91247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82592.actual selector witness) * (LeftAuthority91247.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91251

namespace LeftBound91252
def owner : Owner := ⟨.program ⟨214⟩, ⟨28946⟩⟩
def transferEvent : Nat := 91252
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨28944⟩⟩]⟩ [⟨.result 91248 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91248 .coefficient)
      LeftAuthority91247.bound (LeftAuthority91247.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨28944⟩⟩) (rawTerms := some (Proof.Events356.exact91248RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91247.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91247.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91247.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91252

namespace LeftBound91253
def owner : Owner := ⟨.program ⟨214⟩, ⟨28946⟩⟩
def transferEvent : Nat := 91253
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 82596 .summary) (.transfer 91252) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 82596 .summary)
      LeftBound82595.bound (LeftBound82595.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25375⟩⟩) (rawTerms := some (Proof.Events322.exact82596RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound82595.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91252)
      LeftBound91252.bound (LeftBound91252.actual selector witness) := by
  exact .transfer (LeftBound91252.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound82595.bound LeftBound91252.bound
def bound : CoeffClass := .finite ⟨1292315009023509266432, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound82595.bound, LeftBound91252.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound82595.actual selector witness) * (LeftBound91252.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91253

namespace LeftBound91264
def owner : Owner := ⟨.program ⟨214⟩, ⟨22050⟩⟩
def transferEvent : Nat := 91264
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 91262 .coefficient) (.value (.predecessor 1 91263 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91262 .coefficient)
      LeftAuthority91260.bound (LeftAuthority91260.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91261RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91260.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91263 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority91260.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91260.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91260.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound91264

namespace LeftBound91268
def owner : Owner := ⟨.program ⟨214⟩, ⟨22051⟩⟩
def transferEvent : Nat := 91268
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 91266 .coefficient) (.predecessor 1 91267 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91266 .coefficient)
      LeftBound80009.bound (LeftBound80009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound80009.bound, RecordedBoundRefines] <;> decide)
      (LeftBound80009.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91267 .coefficient)
      LeftBound91264.bound (LeftBound91264.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91265RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91264.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91264.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80009.bound LeftBound91264.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80009.bound, LeftBound91264.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80009.actual selector witness) * (LeftBound91264.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91268

namespace LeftBound91269
def owner : Owner := ⟨.program ⟨214⟩, ⟨22051⟩⟩
def transferEvent : Nat := 91269
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22048⟩⟩]⟩ [⟨.result 91261 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 91261 .coefficient)
      LeftAuthority91260.bound (LeftAuthority91260.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22048⟩⟩) (rawTerms := some (Proof.Events356.exact91261RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91260.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91260.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority91260.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91260.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority91260.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound91269

namespace LeftBound91270
def owner : Owner := ⟨.program ⟨214⟩, ⟨22051⟩⟩
def transferEvent : Nat := 91270
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 80012 .summary) (.transfer 91269) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 80012 .summary)
      LeftBound80010.bound (LeftBound80010.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5541⟩⟩) (rawTerms := some (Proof.Events312.exact80012RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound80010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 91269)
      LeftBound91269.bound (LeftBound91269.actual selector witness) := by
  exact .transfer (LeftBound91269.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound80010.bound LeftBound91269.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound80010.bound, LeftBound91269.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound80010.actual selector witness) * (LeftBound91269.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91270

namespace LeftBound91365
def owner : Owner := ⟨.program ⟨214⟩, ⟨16466⟩⟩
def transferEvent : Nat := 91365
def frameStart : Nat := 91326
def rule : BoundRule := .identity (.predecessor 0 91364 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91364 .coefficient)
      LeftAuthority91362.bound (LeftAuthority91362.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91363RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91362.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91362.derived selector witness)

def rawBound : CoeffClass := LeftAuthority91362.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91362.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority91362.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91365

namespace LeftBound91382
def owner : Owner := ⟨.program ⟨214⟩, ⟨16505⟩⟩
def transferEvent : Nat := 91382
def frameStart : Nat := 91326
def rule : BoundRule := .sum [.predecessor 0 91380 .coefficient, .predecessor 1 91381 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91380 .coefficient)
      LeftBound91365.bound (LeftBound91365.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91365.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91381 .coefficient)
      LeftAuthority91378.bound (LeftAuthority91378.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority91378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound91365.bound, LeftAuthority91378.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91365.bound, LeftAuthority91378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound91365.actual selector witness, LeftAuthority91378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91382

namespace LeftBound91385
def owner : Owner := ⟨.program ⟨214⟩, ⟨16506⟩⟩
def transferEvent : Nat := 91385
def frameStart : Nat := 91326
def rule : BoundRule := .identity (.predecessor 0 91384 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91384 .coefficient)
      LeftBound91382.bound (LeftBound91382.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound91382.derived selector witness)

def rawBound : CoeffClass := LeftBound91382.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91382.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound91382.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound91385

namespace LeftBound91391
def owner : Owner := ⟨.program ⟨214⟩, ⟨16507⟩⟩
def transferEvent : Nat := 91391
def frameStart : Nat := 91326
def rule : BoundRule := .product (.predecessor 0 91389 .coefficient) (.predecessor 1 91390 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91389 .coefficient)
      LeftAuthority91387.bound (LeftAuthority91387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91390 .coefficient)
      LeftBound91385.bound (LeftBound91385.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91386RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91385.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91385.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority91387.bound LeftBound91385.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91387.bound, LeftBound91385.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority91387.actual selector witness) * (LeftBound91385.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91391

namespace LeftBound91399
def owner : Owner := ⟨.program ⟨214⟩, ⟨16508⟩⟩
def transferEvent : Nat := 91399
def frameStart : Nat := 91326
def rule : BoundRule := .sum [.predecessor 0 91397 .coefficient, .predecessor 1 91398 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91397 .coefficient)
      LeftAuthority91395.bound (LeftAuthority91395.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91396RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91395.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91395.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91398 .coefficient)
      LeftBound91391.bound (LeftBound91391.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91393RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91391.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91391.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91395.bound, LeftBound91391.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91395.bound, LeftBound91391.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91395.actual selector witness, LeftBound91391.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91399

namespace LeftBound91403
def owner : Owner := ⟨.program ⟨214⟩, ⟨28945⟩⟩
def transferEvent : Nat := 91403
def frameStart : Nat := 91326
def rule : BoundRule := .product (.predecessor 0 91401 .coefficient) (.predecessor 1 91402 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91401 .coefficient)
      LeftBound91399.bound (LeftBound91399.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91400RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91399.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91399.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91402 .coefficient)
      LeftAuthority91376.bound (LeftAuthority91376.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91377RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91376.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91376.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound91399.bound LeftAuthority91376.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound91399.bound, LeftAuthority91376.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound91399.actual selector witness) * (LeftAuthority91376.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91403

namespace LeftBound91414
def owner : Owner := ⟨.program ⟨214⟩, ⟨17552⟩⟩
def transferEvent : Nat := 91414
def frameStart : Nat := 91326
def rule : BoundRule := .product (.predecessor 0 91412 .coefficient) (.predecessor 1 91413 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91412 .coefficient)
      LeftAuthority91387.bound (LeftAuthority91387.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events356.exact91388RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91387.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91387.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91413 .coefficient)
      LeftAuthority91410.bound (LeftAuthority91410.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91411RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91410.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91410.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority91387.bound LeftAuthority91410.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91387.bound, LeftAuthority91410.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority91387.actual selector witness) * (LeftAuthority91410.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound91414

namespace LeftBound91422
def owner : Owner := ⟨.program ⟨214⟩, ⟨17553⟩⟩
def transferEvent : Nat := 91422
def frameStart : Nat := 91326
def rule : BoundRule := .sum [.predecessor 0 91420 .coefficient, .predecessor 1 91421 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 91420 .coefficient)
      LeftAuthority91418.bound (LeftAuthority91418.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91419RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority91418.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority91418.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 91421 .coefficient)
      LeftBound91414.bound (LeftBound91414.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events357.exact91416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound91414.bound, RecordedBoundRefines] <;> decide)
      (LeftBound91414.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority91418.bound, LeftBound91414.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority91418.bound, LeftBound91414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority91418.actual selector witness, LeftBound91414.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound91422

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
