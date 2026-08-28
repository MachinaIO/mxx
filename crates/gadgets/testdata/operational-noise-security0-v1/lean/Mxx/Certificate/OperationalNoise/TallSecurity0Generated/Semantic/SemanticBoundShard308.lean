import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard018
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard238
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard239
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic.SemanticBoundShard307

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound46335
def owner : Owner := ⟨.program ⟨214⟩, ⟨30168⟩⟩
def transferEvent : Nat := 46335
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46333 .coefficient) (.predecessor 1 46334 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46333 .coefficient)
      LeftBound46294.bound (LeftBound46294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events180.exact46332RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46334 .coefficient)
      LeftBound5498.bound (LeftBound5498.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events021.exact5499RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound5498.bound, RecordedBoundRefines] <;> decide)
      (LeftBound5498.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46294.bound LeftBound5498.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46294.bound, LeftBound5498.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46294.actual selector witness) * (LeftBound5498.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46335

namespace LeftBound46336
def owner : Owner := ⟨.program ⟨214⟩, ⟨30168⟩⟩
def transferEvent : Nat := 46336
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨6651⟩⟩]⟩ [⟨.result 5495 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 5495 .coefficient)
      LeftAuthority5494.bound (LeftAuthority5494.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨6651⟩⟩) (rawTerms := some (Proof.Events021.exact5495RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority5494.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority5494.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority5494.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority5494.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority5494.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46336

namespace LeftBound46337
def owner : Owner := ⟨.program ⟨214⟩, ⟨30168⟩⟩
def transferEvent : Nat := 46337
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46332 .summary) (.transfer 46336) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46332 .summary)
      LeftBound46331.bound (LeftBound46331.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨30167⟩⟩) (rawTerms := some (Proof.Events180.exact46332RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46331.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46336)
      LeftBound46336.bound (LeftBound46336.actual selector witness) := by
  exact .transfer (LeftBound46336.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46331.bound LeftBound46336.bound
def bound : CoeffClass := .finite ⟨313276371396785701094268180805713920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46331.bound, LeftBound46336.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46331.actual selector witness) * (LeftBound46336.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46337

namespace LeftBound46352
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def transferEvent : Nat := 46352
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46350 .coefficient) (.predecessor 1 46351 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46350 .coefficient)
      LeftBound36319.bound (LeftBound36319.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36323RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36319.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36319.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46351 .coefficient)
      LeftAuthority46348.bound (LeftAuthority46348.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46349RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46348.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36319.bound LeftAuthority46348.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36319.bound, LeftAuthority46348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36319.actual selector witness) * (LeftAuthority46348.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46352

namespace LeftBound46353
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def transferEvent : Nat := 46353
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨30154⟩⟩]⟩ [⟨.result 46349 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46349 .coefficient)
      LeftAuthority46348.bound (LeftAuthority46348.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨30154⟩⟩) (rawTerms := some (Proof.Events181.exact46349RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46348.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46348.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46348.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46348.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46348.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46353

namespace LeftBound46354
def owner : Owner := ⟨.program ⟨214⟩, ⟨30156⟩⟩
def transferEvent : Nat := 46354
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36323 .summary) (.transfer 46353) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36323 .summary)
      LeftBound36322.bound (LeftBound36322.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨25770⟩⟩) (rawTerms := some (Proof.Events141.exact36323RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36322.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46353)
      LeftBound46353.bound (LeftBound46353.actual selector witness) := by
  exact .transfer (LeftBound46353.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound36322.bound LeftBound46353.bound
def bound : CoeffClass := .finite ⟨1292539133473715126272, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36322.bound, LeftBound46353.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound36322.actual selector witness) * (LeftBound46353.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46354

namespace LeftBound46365
def owner : Owner := ⟨.program ⟨214⟩, ⟨22778⟩⟩
def transferEvent : Nat := 46365
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 46363 .coefficient) (.value (.predecessor 1 46364 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46363 .coefficient)
      LeftAuthority46361.bound (LeftAuthority46361.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46362RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46361.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46364 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority46361.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46361.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46361.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound46365

namespace LeftBound46369
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def transferEvent : Nat := 46369
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 46367 .coefficient) (.predecessor 1 46368 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46367 .coefficient)
      LeftBound36134.bound (LeftBound36134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound36134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound36134.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46368 .coefficient)
      LeftBound46365.bound (LeftBound46365.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46366RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46365.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46365.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36134.bound LeftBound46365.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36134.bound, LeftBound46365.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36134.actual selector witness) * (LeftBound46365.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46369

namespace LeftBound46370
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def transferEvent : Nat := 46370
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨214⟩, ⟨22776⟩⟩]⟩ [⟨.result 46362 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 46362 .coefficient)
      LeftAuthority46361.bound (LeftAuthority46361.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨214⟩, ⟨22776⟩⟩) (rawTerms := some (Proof.Events181.exact46362RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46361.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46361.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority46361.bound []
def bound : CoeffClass := .finite ⟨136065468, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46361.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := (LeftAuthority46361.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound46370

namespace LeftBound46371
def owner : Owner := ⟨.program ⟨214⟩, ⟨22779⟩⟩
def transferEvent : Nat := 46371
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 36137 .summary) (.transfer 46370) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.result 36137 .summary)
      LeftBound36135.bound (LeftBound36135.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨214⟩, ⟨5553⟩⟩) (rawTerms := some (Proof.Events141.exact36137RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound36135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.transfer 46370)
      LeftBound46370.bound (LeftBound46370.actual selector witness) := by
  exact .transfer (LeftBound46370.derived selector witness)

def rawBound : CoeffClass := productWithFactor 512 LeftBound36135.bound LeftBound46370.bound
def bound : CoeffClass := .finite ⟨1811303510016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound36135.bound, LeftBound46370.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 512 * (LeftBound36135.actual selector witness) * (LeftBound46370.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 16) (rightRows := 16) (rightColumns := 14) (ringDimension := 32) (factor := 512) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46371

namespace LeftBound46466
def owner : Owner := ⟨.program ⟨214⟩, ⟨17020⟩⟩
def transferEvent : Nat := 46466
def frameStart : Nat := 46427
def rule : BoundRule := .identity (.predecessor 0 46465 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46465 .coefficient)
      LeftAuthority46463.bound (LeftAuthority46463.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46464RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46463.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46463.derived selector witness)

def rawBound : CoeffClass := LeftAuthority46463.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46463.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftAuthority46463.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46466

namespace LeftBound46483
def owner : Owner := ⟨.program ⟨214⟩, ⟨17059⟩⟩
def transferEvent : Nat := 46483
def frameStart : Nat := 46427
def rule : BoundRule := .sum [.predecessor 0 46481 .coefficient, .predecessor 1 46482 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46481 .coefficient)
      LeftBound46466.bound (LeftBound46466.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46466.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46482 .coefficient)
      LeftAuthority46479.bound (LeftAuthority46479.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority46479.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound46466.bound, LeftAuthority46479.bound]
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46466.bound, LeftAuthority46479.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftBound46466.actual selector witness, LeftAuthority46479.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46483

namespace LeftBound46486
def owner : Owner := ⟨.program ⟨214⟩, ⟨17060⟩⟩
def transferEvent : Nat := 46486
def frameStart : Nat := 46427
def rule : BoundRule := .identity (.predecessor 0 46485 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46485 .coefficient)
      LeftBound46483.bound (LeftBound46483.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound46483.derived selector witness)

def rawBound : CoeffClass := LeftBound46483.bound
def bound : CoeffClass := .finite ⟨60, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46483.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := LeftBound46483.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound46486

namespace LeftBound46492
def owner : Owner := ⟨.program ⟨214⟩, ⟨17061⟩⟩
def transferEvent : Nat := 46492
def frameStart : Nat := 46427
def rule : BoundRule := .product (.predecessor 0 46490 .coefficient) (.predecessor 1 46491 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46490 .coefficient)
      LeftAuthority46488.bound (LeftAuthority46488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46489RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46488.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46488.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46491 .coefficient)
      LeftBound46486.bound (LeftBound46486.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46487RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46486.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46486.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32 LeftAuthority46488.bound LeftBound46486.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46488.bound, LeftBound46486.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 32 * (LeftAuthority46488.actual selector witness) * (LeftBound46486.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 1) (rightColumns := 1) (ringDimension := 32) (factor := 32) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46492

namespace LeftBound46500
def owner : Owner := ⟨.program ⟨214⟩, ⟨17062⟩⟩
def transferEvent : Nat := 46500
def frameStart : Nat := 46427
def rule : BoundRule := .sum [.predecessor 0 46498 .coefficient, .predecessor 1 46499 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46498 .coefficient)
      LeftAuthority46496.bound (LeftAuthority46496.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46497RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46496.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46496.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46499 .coefficient)
      LeftBound46492.bound (LeftBound46492.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46494RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46492.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46492.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority46496.bound, LeftBound46492.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority46496.bound, LeftBound46492.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := [LeftAuthority46496.actual selector witness, LeftBound46492.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound46500

namespace LeftBound46504
def owner : Owner := ⟨.program ⟨214⟩, ⟨30155⟩⟩
def transferEvent : Nat := 46504
def frameStart : Nat := 46427
def rule : BoundRule := .product (.predecessor 0 46502 .coefficient) (.predecessor 1 46503 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 0 46502 .coefficient)
      LeftBound46500.bound (LeftBound46500.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46501RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46500.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46500.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundInputAt history owner (.predecessor 1 46503 .coefficient)
      LeftAuthority46477.bound (LeftAuthority46477.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events181.exact46478RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority46477.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority46477.derived selector witness)

def rawBound : CoeffClass := productWithFactor 448 LeftBound46500.bound LeftAuthority46477.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46500.bound, LeftAuthority46477.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) : Nat := 448 * (LeftBound46500.actual selector witness) * (LeftAuthority46477.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 100418593683253592432016548326729029359133068138294319235841) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 14) (rightRows := 14) (rightColumns := 14) (ringDimension := 32) (factor := 448) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound46504

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
