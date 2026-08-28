import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard441
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard453

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound76238
def owner : Owner := ⟨.program ⟨214⟩, ⟨29367⟩⟩
def transferEvent : Nat := 76238
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76236 .coefficient) (.predecessor 1 76237 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76236 .coefficient)
      LeftBound67015.bound (LeftBound67015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events261.exact67019RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound67015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound67015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76237 .coefficient)
      LeftAuthority76234.bound (LeftAuthority76234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67015.bound LeftAuthority76234.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67015.bound, LeftAuthority76234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67015.actual selector witness) * (LeftAuthority76234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76238

namespace LeftBound76239
def owner : Owner := ⟨.program ⟨214⟩, ⟨29367⟩⟩
def transferEvent : Nat := 76239
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨29365⟩⟩]⟩ [⟨.result 76235 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76235 .coefficient)
      LeftAuthority76234.bound (LeftAuthority76234.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨29365⟩⟩) (rawTerms := some (Proof.Events297.exact76235RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76234.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76234.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76234.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76239

namespace LeftBound76240
def owner : Owner := ⟨.program ⟨214⟩, ⟨29367⟩⟩
def transferEvent : Nat := 76240
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 67019 .summary) (.transfer 76239) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 67019 .summary)
      LeftBound67018.bound (LeftBound67018.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25524⟩⟩) (rawTerms := some (Proof.Events261.exact67019RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound67018.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76239)
      LeftBound76239.bound (LeftBound76239.actual selector witness) := by
  exact .transfer (LeftBound76239.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound67018.bound LeftBound76239.bound
def bound : CoeffClass := .finite ⟨1292382246358571024384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound67018.bound, LeftBound76239.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound67018.actual selector witness) * (LeftBound76239.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76240

namespace LeftBound76251
def owner : Owner := ⟨.program ⟨214⟩, ⟨22334⟩⟩
def transferEvent : Nat := 76251
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 76249 .coefficient) (.value (.predecessor 1 76250 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76249 .coefficient)
      LeftAuthority76247.bound (LeftAuthority76247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76247.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76250 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority76247.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76247.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76247.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound76251

namespace LeftBound76255
def owner : Owner := ⟨.program ⟨214⟩, ⟨22335⟩⟩
def transferEvent : Nat := 76255
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 76253 .coefficient) (.predecessor 1 76254 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76253 .coefficient)
      LeftBound65384.bound (LeftBound65384.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65384.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76254 .coefficient)
      LeftBound76251.bound (LeftBound76251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events297.exact76252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76251.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65384.bound LeftBound76251.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65384.bound, LeftBound76251.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65384.actual selector witness) * (LeftBound76251.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76255

namespace LeftBound76256
def owner : Owner := ⟨.program ⟨214⟩, ⟨22335⟩⟩
def transferEvent : Nat := 76256
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22332⟩⟩]⟩ [⟨.result 76248 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 76248 .coefficient)
      LeftAuthority76247.bound (LeftAuthority76247.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22332⟩⟩) (rawTerms := some (Proof.Events297.exact76248RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76247.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76247.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority76247.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority76247.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound76256

namespace LeftBound76257
def owner : Owner := ⟨.program ⟨214⟩, ⟨22335⟩⟩
def transferEvent : Nat := 76257
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65387 .summary) (.transfer 76256) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 65387 .summary)
      LeftBound65385.bound (LeftBound65385.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5535⟩⟩) (rawTerms := some (Proof.Events255.exact65387RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65385.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 76256)
      LeftBound76256.bound (LeftBound76256.actual selector witness) := by
  exact .transfer (LeftBound76256.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound65385.bound LeftBound76256.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65385.bound, LeftBound76256.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound65385.actual selector witness) * (LeftBound76256.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76257

namespace LeftBound76352
def owner : Owner := ⟨.program ⟨214⟩, ⟨16630⟩⟩
def transferEvent : Nat := 76352
def frameStart : Nat := 76313
def rule : BoundRule := .identity (.predecessor 0 76351 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76351 .coefficient)
      LeftAuthority76349.bound (LeftAuthority76349.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76350RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76349.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76349.derived selector witness)

def rawBound : CoeffClass := LeftAuthority76349.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76349.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority76349.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76352

namespace LeftBound76369
def owner : Owner := ⟨.program ⟨214⟩, ⟨16704⟩⟩
def transferEvent : Nat := 76369
def frameStart : Nat := 76313
def rule : BoundRule := .sum [.predecessor 0 76367 .coefficient, .predecessor 1 76368 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76367 .coefficient)
      LeftBound76352.bound (LeftBound76352.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76352.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76368 .coefficient)
      LeftAuthority76365.bound (LeftAuthority76365.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority76365.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76352.bound, LeftAuthority76365.bound]
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76352.bound, LeftAuthority76365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76352.actual selector witness, LeftAuthority76365.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76369

namespace LeftBound76372
def owner : Owner := ⟨.program ⟨214⟩, ⟨16705⟩⟩
def transferEvent : Nat := 76372
def frameStart : Nat := 76313
def rule : BoundRule := .identity (.predecessor 0 76371 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76371 .coefficient)
      LeftBound76369.bound (LeftBound76369.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound76369.derived selector witness)

def rawBound : CoeffClass := LeftBound76369.bound
def bound : CoeffClass := .finite ⟨46, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76369.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound76369.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound76372

namespace LeftBound76378
def owner : Owner := ⟨.program ⟨214⟩, ⟨16706⟩⟩
def transferEvent : Nat := 76378
def frameStart : Nat := 76313
def rule : BoundRule := .product (.predecessor 0 76376 .coefficient) (.predecessor 1 76377 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76376 .coefficient)
      LeftAuthority76374.bound (LeftAuthority76374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76374.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76377 .coefficient)
      LeftBound76372.bound (LeftBound76372.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76373RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76372.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76372.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority76374.bound LeftBound76372.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76374.bound, LeftBound76372.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority76374.actual selector witness) * (LeftBound76372.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76378

namespace LeftBound76386
def owner : Owner := ⟨.program ⟨214⟩, ⟨16707⟩⟩
def transferEvent : Nat := 76386
def frameStart : Nat := 76313
def rule : BoundRule := .sum [.predecessor 0 76384 .coefficient, .predecessor 1 76385 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76384 .coefficient)
      LeftAuthority76382.bound (LeftAuthority76382.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76383RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76382.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76382.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76385 .coefficient)
      LeftBound76378.bound (LeftBound76378.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76380RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76378.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76378.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76382.bound, LeftBound76378.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76382.bound, LeftBound76378.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76382.actual selector witness, LeftBound76378.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76386

namespace LeftBound76390
def owner : Owner := ⟨.program ⟨214⟩, ⟨29366⟩⟩
def transferEvent : Nat := 76390
def frameStart : Nat := 76313
def rule : BoundRule := .product (.predecessor 0 76388 .coefficient) (.predecessor 1 76389 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76388 .coefficient)
      LeftBound76386.bound (LeftBound76386.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76387RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76386.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76386.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76389 .coefficient)
      LeftAuthority76363.bound (LeftAuthority76363.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76364RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76363.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76363.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound76386.bound LeftAuthority76363.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76386.bound, LeftAuthority76363.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound76386.actual selector witness) * (LeftAuthority76363.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76390

namespace LeftBound76401
def owner : Owner := ⟨.program ⟨214⟩, ⟨17716⟩⟩
def transferEvent : Nat := 76401
def frameStart : Nat := 76313
def rule : BoundRule := .product (.predecessor 0 76399 .coefficient) (.predecessor 1 76400 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76399 .coefficient)
      LeftAuthority76374.bound (LeftAuthority76374.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76375RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76374.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76374.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76400 .coefficient)
      LeftAuthority76397.bound (LeftAuthority76397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76397.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76397.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority76374.bound LeftAuthority76397.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76374.bound, LeftAuthority76397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 1 * (LeftAuthority76374.actual selector witness) * (LeftAuthority76397.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound76401

namespace LeftBound76409
def owner : Owner := ⟨.program ⟨214⟩, ⟨17717⟩⟩
def transferEvent : Nat := 76409
def frameStart : Nat := 76313
def rule : BoundRule := .sum [.predecessor 0 76407 .coefficient, .predecessor 1 76408 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76407 .coefficient)
      LeftAuthority76405.bound (LeftAuthority76405.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76406RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority76405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority76405.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76408 .coefficient)
      LeftBound76401.bound (LeftBound76401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76401.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority76405.bound, LeftBound76401.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority76405.bound, LeftBound76401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority76405.actual selector witness, LeftBound76401.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76409

namespace LeftBound76413
def owner : Owner := ⟨.program ⟨214⟩, ⟨29371⟩⟩
def transferEvent : Nat := 76413
def frameStart : Nat := 76313
def rule : BoundRule := .sum [.predecessor 0 76411 .coefficient, .predecessor 1 76412 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 76411 .coefficient)
      LeftBound76409.bound (LeftBound76409.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76410RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76409.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 76412 .coefficient)
      LeftBound76390.bound (LeftBound76390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events298.exact76395RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound76390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound76390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound76409.bound, LeftBound76390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound76409.bound, LeftBound76390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound76409.actual selector witness, LeftBound76390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound76413

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
