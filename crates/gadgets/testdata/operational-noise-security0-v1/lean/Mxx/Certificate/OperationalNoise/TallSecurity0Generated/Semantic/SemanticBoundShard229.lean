import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard137
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard199
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard200
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard228

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound35316
def owner : Owner := ⟨.program ⟨214⟩, ⟨26600⟩⟩
def transferEvent : Nat := 35316
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 35311 .summary) (.transfer 35315) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35311 .summary)
      LeftBound35310.bound (LeftBound35310.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨26599⟩⟩) (rawTerms := some (Proof.Events137.exact35311RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound35310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35315)
      LeftBound35315.bound (LeftBound35315.actual selector witness) := by
  exact .transfer (LeftBound35315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35310.bound LeftBound35315.bound
def bound : CoeffClass := .finite ⟨4741295067215179835091451904, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35310.bound, LeftBound35315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35310.actual selector witness) * (LeftBound35315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35316

namespace LeftBound35331
def owner : Owner := ⟨.program ⟨214⟩, ⟨26389⟩⟩
def transferEvent : Nat := 35331
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35329 .coefficient) (.predecessor 1 35330 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35329 .coefficient)
      LeftBound29888.bound (LeftBound29888.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events116.exact29892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound29888.bound, RecordedBoundRefines] <;> decide)
      (LeftBound29888.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35330 .coefficient)
      LeftAuthority35327.bound (LeftAuthority35327.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35328RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35327.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29888.bound LeftAuthority35327.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29888.bound, LeftAuthority35327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29888.actual selector witness) * (LeftAuthority35327.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35331

namespace LeftBound35332
def owner : Owner := ⟨.program ⟨214⟩, ⟨26389⟩⟩
def transferEvent : Nat := 35332
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨26387⟩⟩]⟩ [⟨.result 35328 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35328 .coefficient)
      LeftAuthority35327.bound (LeftAuthority35327.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨26387⟩⟩) (rawTerms := some (Proof.Events138.exact35328RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35327.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35327.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35327.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35327.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35327.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35332

namespace LeftBound35333
def owner : Owner := ⟨.program ⟨214⟩, ⟨26389⟩⟩
def transferEvent : Nat := 35333
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 29892 .summary) (.transfer 35332) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 29892 .summary)
      LeftBound29891.bound (LeftBound29891.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨24928⟩⟩) (rawTerms := some (Proof.Events116.exact29892RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound29891.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35332)
      LeftBound35332.bound (LeftBound35332.actual selector witness) := by
  exact .transfer (LeftBound35332.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound29891.bound LeftBound35332.bound
def bound : CoeffClass := .finite ⟨1291889172568118132736, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound29891.bound, LeftBound35332.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound29891.actual selector witness) * (LeftBound35332.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35333

namespace LeftBound35344
def owner : Owner := ⟨.program ⟨214⟩, ⟨20334⟩⟩
def transferEvent : Nat := 35344
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 35342 .coefficient) (.value (.predecessor 1 35343 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35342 .coefficient)
      LeftAuthority35340.bound (LeftAuthority35340.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35341RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35340.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35343 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority35340.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35340.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35340.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound35344

namespace LeftBound35348
def owner : Owner := ⟨.program ⟨214⟩, ⟨20335⟩⟩
def transferEvent : Nat := 35348
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 35346 .coefficient) (.predecessor 1 35347 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35346 .coefficient)
      LeftBound21509.bound (LeftBound21509.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21509.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21509.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35347 .coefficient)
      LeftBound35344.bound (LeftBound35344.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35345RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35344.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35344.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21509.bound LeftBound35344.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21509.bound, LeftBound35344.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21509.actual selector witness) * (LeftBound35344.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35348

namespace LeftBound35349
def owner : Owner := ⟨.program ⟨214⟩, ⟨20335⟩⟩
def transferEvent : Nat := 35349
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨20332⟩⟩]⟩ [⟨.result 35341 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 35341 .coefficient)
      LeftAuthority35340.bound (LeftAuthority35340.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨20332⟩⟩) (rawTerms := some (Proof.Events138.exact35341RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35340.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35340.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority35340.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35340.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority35340.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound35349

namespace LeftBound35350
def owner : Owner := ⟨.program ⟨214⟩, ⟨20335⟩⟩
def transferEvent : Nat := 35350
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 21512 .summary) (.transfer 35349) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 21512 .summary)
      LeftBound21510.bound (LeftBound21510.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5559⟩⟩) (rawTerms := some (Proof.Events084.exact21512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound21510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 35349)
      LeftBound35349.bound (LeftBound35349.actual selector witness) := by
  exact .transfer (LeftBound35349.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound21510.bound LeftBound35349.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21510.bound, LeftBound35349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound21510.actual selector witness) * (LeftBound35349.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35350

namespace LeftBound35445
def owner : Owner := ⟨.program ⟨214⟩, ⟨14805⟩⟩
def transferEvent : Nat := 35445
def frameStart : Nat := 35406
def rule : BoundRule := .identity (.predecessor 0 35444 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35444 .coefficient)
      LeftAuthority35442.bound (LeftAuthority35442.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35443RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35442.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35442.derived selector witness)

def rawBound : CoeffClass := LeftAuthority35442.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35442.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority35442.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound35445

namespace LeftBound35462
def owner : Owner := ⟨.program ⟨214⟩, ⟨14844⟩⟩
def transferEvent : Nat := 35462
def frameStart : Nat := 35406
def rule : BoundRule := .sum [.predecessor 0 35460 .coefficient, .predecessor 1 35461 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35460 .coefficient)
      LeftBound35445.bound (LeftBound35445.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound35445.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35461 .coefficient)
      LeftAuthority35458.bound (LeftAuthority35458.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority35458.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound35445.bound, LeftAuthority35458.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35445.bound, LeftAuthority35458.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound35445.actual selector witness, LeftAuthority35458.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35462

namespace LeftBound35465
def owner : Owner := ⟨.program ⟨214⟩, ⟨14845⟩⟩
def transferEvent : Nat := 35465
def frameStart : Nat := 35406
def rule : BoundRule := .identity (.predecessor 0 35464 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35464 .coefficient)
      LeftBound35462.bound (LeftBound35462.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound35462.derived selector witness)

def rawBound : CoeffClass := LeftBound35462.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35462.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound35462.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound35465

namespace LeftBound35471
def owner : Owner := ⟨.program ⟨214⟩, ⟨14846⟩⟩
def transferEvent : Nat := 35471
def frameStart : Nat := 35406
def rule : BoundRule := .product (.predecessor 0 35469 .coefficient) (.predecessor 1 35470 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35469 .coefficient)
      LeftAuthority35467.bound (LeftAuthority35467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35470 .coefficient)
      LeftBound35465.bound (LeftBound35465.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35466RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35465.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35465.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority35467.bound LeftBound35465.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35467.bound, LeftBound35465.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority35467.actual selector witness) * (LeftBound35465.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35471

namespace LeftBound35479
def owner : Owner := ⟨.program ⟨214⟩, ⟨14847⟩⟩
def transferEvent : Nat := 35479
def frameStart : Nat := 35406
def rule : BoundRule := .sum [.predecessor 0 35477 .coefficient, .predecessor 1 35478 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35477 .coefficient)
      LeftAuthority35475.bound (LeftAuthority35475.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35476RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35475.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35475.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35478 .coefficient)
      LeftBound35471.bound (LeftBound35471.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35473RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35471.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35471.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35475.bound, LeftBound35471.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35475.bound, LeftBound35471.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35475.actual selector witness, LeftBound35471.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35479

namespace LeftBound35483
def owner : Owner := ⟨.program ⟨214⟩, ⟨26388⟩⟩
def transferEvent : Nat := 35483
def frameStart : Nat := 35406
def rule : BoundRule := .product (.predecessor 0 35481 .coefficient) (.predecessor 1 35482 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35481 .coefficient)
      LeftBound35479.bound (LeftBound35479.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35480RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35479.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35479.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35482 .coefficient)
      LeftAuthority35456.bound (LeftAuthority35456.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35457RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35456.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35456.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound35479.bound LeftAuthority35456.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound35479.bound, LeftAuthority35456.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound35479.actual selector witness) * (LeftAuthority35456.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35483

namespace LeftBound35494
def owner : Owner := ⟨.program ⟨214⟩, ⟨14904⟩⟩
def transferEvent : Nat := 35494
def frameStart : Nat := 35406
def rule : BoundRule := .product (.predecessor 0 35492 .coefficient) (.predecessor 1 35493 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35492 .coefficient)
      LeftAuthority35467.bound (LeftAuthority35467.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35468RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35467.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35467.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35493 .coefficient)
      LeftAuthority35490.bound (LeftAuthority35490.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35491RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35490.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35490.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority35467.bound LeftAuthority35490.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35467.bound, LeftAuthority35490.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority35467.actual selector witness) * (LeftAuthority35490.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound35494

namespace LeftBound35502
def owner : Owner := ⟨.program ⟨214⟩, ⟨14905⟩⟩
def transferEvent : Nat := 35502
def frameStart : Nat := 35406
def rule : BoundRule := .sum [.predecessor 0 35500 .coefficient, .predecessor 1 35501 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 35500 .coefficient)
      LeftAuthority35498.bound (LeftAuthority35498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35498.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35498.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 35501 .coefficient)
      LeftBound35494.bound (LeftBound35494.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events138.exact35496RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound35494.bound, RecordedBoundRefines] <;> decide)
      (LeftBound35494.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority35498.bound, LeftBound35494.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority35498.bound, LeftBound35494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority35498.actual selector witness, LeftBound35494.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound35502

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
