import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard245
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard246

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound37283
def owner : Owner := ⟨.program ⟨214⟩, ⟨25616⟩⟩
def transferEvent : Nat := 37283
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 37281 .coefficient, .predecessor 1 37282 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37281 .coefficient)
      LeftBound37104.bound (LeftBound37104.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37104.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37282 .coefficient)
      LeftBound37087.bound (LeftBound37087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37104.bound, LeftBound37087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37104.bound, LeftBound37087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37104.actual selector witness, LeftBound37087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37283

namespace LeftBound37286
def owner : Owner := ⟨.program ⟨214⟩, ⟨25616⟩⟩
def transferEvent : Nat := 37286
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 37280 .summary, .result 37094 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37280 .summary)
      LeftBound37106.bound (LeftBound37106.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨20115⟩⟩) (rawTerms := some (Proof.Events145.exact37280RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37106.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37094 .summary)
      LeftBound37089.bound (LeftBound37089.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25615⟩⟩) (rawTerms := some (Proof.Events144.exact37094RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37089.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37106.bound, LeftBound37089.bound]
def bound : CoeffClass := .finite ⟨352164536528896, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37106.bound, LeftBound37089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37106.actual selector witness, LeftBound37089.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37286

namespace LeftBound37290
def owner : Owner := ⟨.program ⟨214⟩, ⟨29630⟩⟩
def transferEvent : Nat := 37290
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37288 .coefficient) (.predecessor 1 37289 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37288 .coefficient)
      LeftBound37283.bound (LeftBound37283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37283.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37289 .coefficient)
      LeftAuthority37009.bound (LeftAuthority37009.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events144.exact37010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37009.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37283.bound LeftAuthority37009.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37283.bound, LeftAuthority37009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37283.actual selector witness) * (LeftAuthority37009.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37290

namespace LeftBound37291
def owner : Owner := ⟨.program ⟨214⟩, ⟨29630⟩⟩
def transferEvent : Nat := 37291
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29628⟩⟩]⟩ [⟨.result 37010 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37010 .coefficient)
      LeftAuthority37009.bound (LeftAuthority37009.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29628⟩⟩) (rawTerms := some (Proof.Events144.exact37010RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37009.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37009.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37009.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37009.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37009.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37291

namespace LeftBound37292
def owner : Owner := ⟨.program ⟨214⟩, ⟨29630⟩⟩
def transferEvent : Nat := 37292
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 37287 .summary) (.transfer 37291) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37287 .summary)
      LeftBound37286.bound (LeftBound37286.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25616⟩⟩) (rawTerms := some (Proof.Events145.exact37287RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound37286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37291)
      LeftBound37291.bound (LeftBound37291.actual selector witness) := by
  exact .transfer (LeftBound37291.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37286.bound LeftBound37291.bound
def bound : CoeffClass := .finite ⟨1292449483693632782336, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37286.bound, LeftBound37291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37286.actual selector witness) * (LeftBound37291.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37292

namespace LeftBound37303
def owner : Owner := ⟨.program ⟨214⟩, ⟨22562⟩⟩
def transferEvent : Nat := 37303
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 37301 .coefficient) (.value (.predecessor 1 37302 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37301 .coefficient)
      LeftAuthority37299.bound (LeftAuthority37299.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37300RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37299.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37299.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37302 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority37299.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37299.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37299.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound37303

namespace LeftBound37307
def owner : Owner := ⟨.program ⟨214⟩, ⟨22563⟩⟩
def transferEvent : Nat := 37307
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 37305 .coefficient) (.predecessor 1 37306 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37305 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37306 .coefficient)
      LeftBound37303.bound (LeftBound37303.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events145.exact37304RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37303.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37303.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound37303.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound37303.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound37303.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37307

namespace LeftBound37308
def owner : Owner := ⟨.program ⟨214⟩, ⟨22563⟩⟩
def transferEvent : Nat := 37308
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22560⟩⟩]⟩ [⟨.result 37300 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 37300 .coefficient)
      LeftAuthority37299.bound (LeftAuthority37299.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22560⟩⟩) (rawTerms := some (Proof.Events145.exact37300RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37299.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37299.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority37299.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37299.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority37299.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound37308

namespace LeftBound37309
def owner : Owner := ⟨.program ⟨214⟩, ⟨22563⟩⟩
def transferEvent : Nat := 37309
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 37308) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 37308)
      LeftBound37308.bound (LeftBound37308.actual selector witness) := by
  exact .transfer (LeftBound37308.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound37308.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound37308.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound37308.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37309

namespace LeftBound37404
def owner : Owner := ⟨.program ⟨214⟩, ⟨16761⟩⟩
def transferEvent : Nat := 37404
def frameStart : Nat := 37365
def rule : BoundRule := .identity (.predecessor 0 37403 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37403 .coefficient)
      LeftAuthority37401.bound (LeftAuthority37401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37402RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37401.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37401.derived selector witness)

def rawBound : CoeffClass := LeftAuthority37401.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority37401.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37404

namespace LeftBound37421
def owner : Owner := ⟨.program ⟨214⟩, ⟨16835⟩⟩
def transferEvent : Nat := 37421
def frameStart : Nat := 37365
def rule : BoundRule := .sum [.predecessor 0 37419 .coefficient, .predecessor 1 37420 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37419 .coefficient)
      LeftBound37404.bound (LeftBound37404.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37404.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37420 .coefficient)
      LeftAuthority37417.bound (LeftAuthority37417.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority37417.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound37404.bound, LeftAuthority37417.bound]
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37404.bound, LeftAuthority37417.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound37404.actual selector witness, LeftAuthority37417.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37421

namespace LeftBound37424
def owner : Owner := ⟨.program ⟨214⟩, ⟨16836⟩⟩
def transferEvent : Nat := 37424
def frameStart : Nat := 37365
def rule : BoundRule := .identity (.predecessor 0 37423 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37423 .coefficient)
      LeftBound37421.bound (LeftBound37421.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound37421.derived selector witness)

def rawBound : CoeffClass := LeftBound37421.bound
def bound : CoeffClass := .finite ⟨52, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37421.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound37421.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound37424

namespace LeftBound37430
def owner : Owner := ⟨.program ⟨214⟩, ⟨16837⟩⟩
def transferEvent : Nat := 37430
def frameStart : Nat := 37365
def rule : BoundRule := .product (.predecessor 0 37428 .coefficient) (.predecessor 1 37429 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37428 .coefficient)
      LeftAuthority37426.bound (LeftAuthority37426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37429 .coefficient)
      LeftBound37424.bound (LeftBound37424.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37425RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37424.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37424.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority37426.bound LeftBound37424.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37426.bound, LeftBound37424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority37426.actual selector witness) * (LeftBound37424.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37430

namespace LeftBound37438
def owner : Owner := ⟨.program ⟨214⟩, ⟨16838⟩⟩
def transferEvent : Nat := 37438
def frameStart : Nat := 37365
def rule : BoundRule := .sum [.predecessor 0 37436 .coefficient, .predecessor 1 37437 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37436 .coefficient)
      LeftAuthority37434.bound (LeftAuthority37434.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37435RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37434.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37434.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37437 .coefficient)
      LeftBound37430.bound (LeftBound37430.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37432RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37430.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37430.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority37434.bound, LeftBound37430.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37434.bound, LeftBound37430.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority37434.actual selector witness, LeftBound37430.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound37438

namespace LeftBound37442
def owner : Owner := ⟨.program ⟨214⟩, ⟨29629⟩⟩
def transferEvent : Nat := 37442
def frameStart : Nat := 37365
def rule : BoundRule := .product (.predecessor 0 37440 .coefficient) (.predecessor 1 37441 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37440 .coefficient)
      LeftBound37438.bound (LeftBound37438.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37439RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound37438.bound, RecordedBoundRefines] <;> decide)
      (LeftBound37438.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37441 .coefficient)
      LeftAuthority37415.bound (LeftAuthority37415.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37416RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37415.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37415.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound37438.bound LeftAuthority37415.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound37438.bound, LeftAuthority37415.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound37438.actual selector witness) * (LeftAuthority37415.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37442

namespace LeftBound37453
def owner : Owner := ⟨.program ⟨214⟩, ⟨16805⟩⟩
def transferEvent : Nat := 37453
def frameStart : Nat := 37365
def rule : BoundRule := .product (.predecessor 0 37451 .coefficient) (.predecessor 1 37452 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 37451 .coefficient)
      LeftAuthority37426.bound (LeftAuthority37426.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37427RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37426.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37426.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 37452 .coefficient)
      LeftAuthority37449.bound (LeftAuthority37449.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events146.exact37450RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority37449.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority37449.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority37426.bound LeftAuthority37449.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority37426.bound, LeftAuthority37449.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority37426.actual selector witness) * (LeftAuthority37449.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound37453

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
