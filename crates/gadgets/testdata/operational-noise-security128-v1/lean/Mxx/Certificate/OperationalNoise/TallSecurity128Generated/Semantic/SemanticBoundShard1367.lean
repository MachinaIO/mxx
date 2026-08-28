import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1291
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1310
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1366

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound204246
def owner : Owner := ⟨.program ⟨257⟩, ⟨39356⟩⟩
def transferEvent : Nat := 204246
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 204244 .coefficient, .predecessor 1 204245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204244 .coefficient)
      LeftBound204075.bound (LeftBound204075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204245 .coefficient)
      LeftBound204058.bound (LeftBound204058.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204065RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204058.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204058.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound204075.bound, LeftBound204058.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204075.bound, LeftBound204058.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound204075.actual selector witness, LeftBound204058.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204246

namespace LeftBound204249
def owner : Owner := ⟨.program ⟨257⟩, ⟨39356⟩⟩
def transferEvent : Nat := 204249
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 204243 .summary, .result 204065 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204243 .summary)
      LeftBound204077.bound (LeftBound204077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38215⟩⟩) (rawTerms := some (Proof.Events797.exact204243RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204077.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204065 .summary)
      LeftBound204060.bound (LeftBound204060.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39355⟩⟩) (rawTerms := some (Proof.Events797.exact204065RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204060.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound204077.bound, LeftBound204060.bound]
def bound : CoeffClass := .finite ⟨32192736221397454434328420548608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204077.bound, LeftBound204060.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound204077.actual selector witness, LeftBound204060.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204249

namespace LeftBound204253
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def transferEvent : Nat := 204253
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 204251 .coefficient) (.predecessor 1 204252 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204251 .coefficient)
      LeftBound204246.bound (LeftBound204246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204250RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204252 .coefficient)
      LeftBound15621.bound (LeftBound15621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound204246.bound LeftBound15621.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204246.bound, LeftBound15621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound204246.actual selector witness) * (LeftBound15621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204253

namespace LeftBound204254
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def transferEvent : Nat := 204254
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩ [⟨.result 15618 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15618 .coefficient)
      LeftAuthority15617.bound (LeftAuthority15617.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7161⟩⟩) (rawTerms := some (Proof.Events061.exact15618RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15617.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15617.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15617.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15617.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15617.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound204254

namespace LeftBound204255
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def transferEvent : Nat := 204255
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 204250 .summary) (.transfer 204254) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204250 .summary)
      LeftBound204249.bound (LeftBound204249.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨39356⟩⟩) (rawTerms := some (Proof.Events797.exact204250RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound204249.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 204254)
      LeftBound204254.bound (LeftBound204254.actual selector witness) := by
  exact .transfer (LeftBound204254.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound204249.bound LeftBound204254.bound
def bound : CoeffClass := .finite ⟨345666873099141705532726864949014345809920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204249.bound, LeftBound204254.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound204249.actual selector witness) * (LeftBound204254.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204255

namespace LeftBound204270
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def transferEvent : Nat := 204270
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 204268 .coefficient) (.predecessor 1 204269 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204268 .coefficient)
      LeftBound195587.bound (LeftBound195587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events764.exact195591RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound195587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound195587.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204269 .coefficient)
      LeftAuthority204266.bound (LeftAuthority204266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound195587.bound LeftAuthority204266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195587.bound, LeftAuthority204266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound195587.actual selector witness) * (LeftAuthority204266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204270

namespace LeftBound204271
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def transferEvent : Nat := 204271
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩ [⟨.result 204267 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204267 .coefficient)
      LeftAuthority204266.bound (LeftAuthority204266.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨36673⟩⟩) (rawTerms := some (Proof.Events797.exact204267RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204266.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority204266.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority204266.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound204271

namespace LeftBound204272
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def transferEvent : Nat := 204272
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 195591 .summary) (.transfer 204271) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 195591 .summary)
      LeftBound195590.bound (LeftBound195590.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36283⟩⟩) (rawTerms := some (Proof.Events764.exact195591RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound195590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 204271)
      LeftBound204271.bound (LeftBound204271.actual selector witness) := by
  exact .transfer (LeftBound204271.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound195590.bound LeftBound204271.bound
def bound : CoeffClass := .finite ⟨32192539770951564984245676933120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound195590.bound, LeftBound204271.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound195590.actual selector witness) * (LeftBound204271.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204272

namespace LeftBound204283
def owner : Owner := ⟨.program ⟨257⟩, ⟨35534⟩⟩
def transferEvent : Nat := 204283
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 204281 .coefficient) (.value (.predecessor 1 204282 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204281 .coefficient)
      LeftAuthority204279.bound (LeftAuthority204279.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204280RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204279.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204282 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority204279.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204279.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority204279.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound204283

namespace LeftBound204287
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def transferEvent : Nat := 204287
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 204285 .coefficient) (.predecessor 1 204286 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204285 .coefficient)
      LeftBound192992.bound (LeftBound192992.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound192992.bound, RecordedBoundRefines] <;> decide)
      (LeftBound192992.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204286 .coefficient)
      LeftBound204283.bound (LeftBound204283.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events797.exact204284RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204283.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204283.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192992.bound LeftBound204283.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192992.bound, LeftBound204283.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192992.actual selector witness) * (LeftBound204283.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204287

namespace LeftBound204288
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def transferEvent : Nat := 204288
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩ [⟨.result 204280 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 204280 .coefficient)
      LeftAuthority204279.bound (LeftAuthority204279.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35532⟩⟩) (rawTerms := some (Proof.Events797.exact204280RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204279.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204279.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority204279.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204279.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority204279.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound204288

namespace LeftBound204289
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def transferEvent : Nat := 204289
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 192995 .summary) (.transfer 204288) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 192995 .summary)
      LeftBound192993.bound (LeftBound192993.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5909⟩⟩) (rawTerms := some (Proof.Events753.exact192995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound192993.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 204288)
      LeftBound204288.bound (LeftBound204288.actual selector witness) := by
  exact .transfer (LeftBound204288.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound192993.bound LeftBound204288.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound192993.bound, LeftBound204288.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound192993.actual selector witness) * (LeftBound204288.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204289

namespace LeftBound204384
def owner : Owner := ⟨.program ⟨257⟩, ⟨34765⟩⟩
def transferEvent : Nat := 204384
def frameStart : Nat := 204345
def rule : BoundRule := .identity (.predecessor 0 204383 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204383 .coefficient)
      LeftAuthority204381.bound (LeftAuthority204381.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events798.exact204382RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204381.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204381.derived selector witness)

def rawBound : CoeffClass := LeftAuthority204381.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204381.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority204381.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound204384

namespace LeftBound204401
def owner : Owner := ⟨.program ⟨257⟩, ⟨36114⟩⟩
def transferEvent : Nat := 204401
def frameStart : Nat := 204345
def rule : BoundRule := .sum [.predecessor 0 204399 .coefficient, .predecessor 1 204400 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204399 .coefficient)
      LeftBound204384.bound (LeftBound204384.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound204384.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204400 .coefficient)
      LeftAuthority204397.bound (LeftAuthority204397.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority204397.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound204384.bound, LeftAuthority204397.bound]
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204384.bound, LeftAuthority204397.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound204384.actual selector witness, LeftAuthority204397.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound204401

namespace LeftBound204404
def owner : Owner := ⟨.program ⟨257⟩, ⟨36115⟩⟩
def transferEvent : Nat := 204404
def frameStart : Nat := 204345
def rule : BoundRule := .identity (.predecessor 0 204403 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204403 .coefficient)
      LeftBound204401.bound (LeftBound204401.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound204401.derived selector witness)

def rawBound : CoeffClass := LeftBound204401.bound
def bound : CoeffClass := .finite ⟨40, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound204401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound204401.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound204404

namespace LeftBound204410
def owner : Owner := ⟨.program ⟨257⟩, ⟨36116⟩⟩
def transferEvent : Nat := 204410
def frameStart : Nat := 204345
def rule : BoundRule := .product (.predecessor 0 204408 .coefficient) (.predecessor 1 204409 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 204408 .coefficient)
      LeftAuthority204406.bound (LeftAuthority204406.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events798.exact204407RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority204406.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority204406.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 204409 .coefficient)
      LeftBound204404.bound (LeftBound204404.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events798.exact204405RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound204404.bound, RecordedBoundRefines] <;> decide)
      (LeftBound204404.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority204406.bound LeftBound204404.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority204406.bound, LeftBound204404.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority204406.actual selector witness) * (LeftBound204404.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound204410

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
