import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard050
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1600
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1665

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound247260
def owner : Owner := ⟨.program ⟨257⟩, ⟨49978⟩⟩
def transferEvent : Nat := 247260
def frameStart : Nat := 247160
def rule : BoundRule := .sum [.predecessor 0 247258 .coefficient, .predecessor 1 247259 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247258 .coefficient)
      LeftBound247256.bound (LeftBound247256.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247257RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247256.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247256.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247259 .coefficient)
      LeftBound247237.bound (LeftBound247237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247237.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound247256.bound, LeftBound247237.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247256.bound, LeftBound247237.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound247256.actual selector witness, LeftBound247237.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound247260

namespace LeftBound247273
def owner : Owner := ⟨.program ⟨257⟩, ⟨49976⟩⟩
def transferEvent : Nat := 247273
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 247271 .coefficient, .predecessor 1 247272 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247271 .coefficient)
      LeftBound247102.bound (LeftBound247102.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247270RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247102.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247102.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247272 .coefficient)
      LeftBound247085.bound (LeftBound247085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247092RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound247102.bound, LeftBound247085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247102.bound, LeftBound247085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound247102.actual selector witness, LeftBound247085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound247273

namespace LeftBound247276
def owner : Owner := ⟨.program ⟨257⟩, ⟨49976⟩⟩
def transferEvent : Nat := 247276
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 247270 .summary, .result 247092 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247270 .summary)
      LeftBound247104.bound (LeftBound247104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨48855⟩⟩) (rawTerms := some (Proof.Events965.exact247270RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound247104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247092 .summary)
      LeftBound247087.bound (LeftBound247087.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49975⟩⟩) (rawTerms := some (Proof.Events965.exact247092RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound247087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound247104.bound, LeftBound247087.bound]
def bound : CoeffClass := .finite ⟨32194504275408640829496428331008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247104.bound, LeftBound247087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound247104.actual selector witness, LeftBound247087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound247276

namespace LeftBound247280
def owner : Owner := ⟨.program ⟨257⟩, ⟨49977⟩⟩
def transferEvent : Nat := 247280
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 247278 .coefficient) (.predecessor 1 247279 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247278 .coefficient)
      LeftBound247273.bound (LeftBound247273.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247277RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247273.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247273.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247279 .coefficient)
      LeftBound15541.bound (LeftBound15541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15542RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15541.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound247273.bound LeftBound15541.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247273.bound, LeftBound15541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound247273.actual selector witness) * (LeftBound15541.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247280

namespace LeftBound247281
def owner : Owner := ⟨.program ⟨257⟩, ⟨49977⟩⟩
def transferEvent : Nat := 247281
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7147⟩⟩]⟩ [⟨.result 15538 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15538 .coefficient)
      LeftAuthority15537.bound (LeftAuthority15537.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7147⟩⟩) (rawTerms := some (Proof.Events060.exact15538RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15537.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15537.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15537.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15537.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15537.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound247281

namespace LeftBound247282
def owner : Owner := ⟨.program ⟨257⟩, ⟨49977⟩⟩
def transferEvent : Nat := 247282
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 247277 .summary) (.transfer 247281) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247277 .summary)
      LeftBound247276.bound (LeftBound247276.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49976⟩⟩) (rawTerms := some (Proof.Events965.exact247277RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound247276.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 247281)
      LeftBound247281.bound (LeftBound247281.actual selector witness) := by
  exact .transfer (LeftBound247281.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound247276.bound LeftBound247281.bound
def bound : CoeffClass := .finite ⟨345685857434530723496243679576218056785920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247276.bound, LeftBound247281.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound247276.actual selector witness) * (LeftBound247281.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247282

namespace LeftBound247297
def owner : Owner := ⟨.program ⟨257⟩, ⟨47295⟩⟩
def transferEvent : Nat := 247297
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 247295 .coefficient) (.predecessor 1 247296 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247295 .coefficient)
      LeftBound237534.bound (LeftBound237534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events927.exact237538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound237534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound237534.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247296 .coefficient)
      LeftAuthority247293.bound (LeftAuthority247293.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events965.exact247294RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority247293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority247293.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237534.bound LeftAuthority247293.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237534.bound, LeftAuthority247293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237534.actual selector witness) * (LeftAuthority247293.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247297

namespace LeftBound247298
def owner : Owner := ⟨.program ⟨257⟩, ⟨47295⟩⟩
def transferEvent : Nat := 247298
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47293⟩⟩]⟩ [⟨.result 247294 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247294 .coefficient)
      LeftAuthority247293.bound (LeftAuthority247293.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47293⟩⟩) (rawTerms := some (Proof.Events965.exact247294RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority247293.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority247293.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority247293.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority247293.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority247293.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound247298

namespace LeftBound247299
def owner : Owner := ⟨.program ⟨257⟩, ⟨47295⟩⟩
def transferEvent : Nat := 247299
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 237538 .summary) (.transfer 247298) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 237538 .summary)
      LeftBound237537.bound (LeftBound237537.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨46959⟩⟩) (rawTerms := some (Proof.Events927.exact237538RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound237537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 247298)
      LeftBound247298.bound (LeftBound247298.actual selector witness) := by
  exact .transfer (LeftBound247298.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound237537.bound LeftBound247298.bound
def bound : CoeffClass := .finite ⟨32194307824962751379413684715520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound237537.bound, LeftBound247298.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound237537.actual selector witness) * (LeftBound247298.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247299

namespace LeftBound247310
def owner : Owner := ⟨.program ⟨257⟩, ⟨46174⟩⟩
def transferEvent : Nat := 247310
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 247308 .coefficient) (.value (.predecessor 1 247309 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247308 .coefficient)
      LeftAuthority247306.bound (LeftAuthority247306.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events966.exact247307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority247306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority247306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247309 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority247306.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority247306.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority247306.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound247310

namespace LeftBound247314
def owner : Owner := ⟨.program ⟨257⟩, ⟨46175⟩⟩
def transferEvent : Nat := 247314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 247312 .coefficient) (.predecessor 1 247313 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247312 .coefficient)
      LeftBound236867.bound (LeftBound236867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247313 .coefficient)
      LeftBound247310.bound (LeftBound247310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events966.exact247311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound247310.bound, RecordedBoundRefines] <;> decide)
      (LeftBound247310.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236867.bound LeftBound247310.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236867.bound, LeftBound247310.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236867.actual selector witness) * (LeftBound247310.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247314

namespace LeftBound247315
def owner : Owner := ⟨.program ⟨257⟩, ⟨46175⟩⟩
def transferEvent : Nat := 247315
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨46172⟩⟩]⟩ [⟨.result 247307 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 247307 .coefficient)
      LeftAuthority247306.bound (LeftAuthority247306.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨46172⟩⟩) (rawTerms := some (Proof.Events966.exact247307RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority247306.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority247306.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority247306.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority247306.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority247306.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound247315

namespace LeftBound247316
def owner : Owner := ⟨.program ⟨257⟩, ⟨46175⟩⟩
def transferEvent : Nat := 247316
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 247315) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 247315)
      LeftBound247315.bound (LeftBound247315.actual selector witness) := by
  exact .transfer (LeftBound247315.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound247315.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound247315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound247315.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound247316

namespace LeftBound247411
def owner : Owner := ⟨.program ⟨257⟩, ⟨45453⟩⟩
def transferEvent : Nat := 247411
def frameStart : Nat := 247372
def rule : BoundRule := .identity (.predecessor 0 247410 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247410 .coefficient)
      LeftAuthority247408.bound (LeftAuthority247408.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events966.exact247409RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority247408.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority247408.derived selector witness)

def rawBound : CoeffClass := LeftAuthority247408.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority247408.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority247408.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound247411

namespace LeftBound247428
def owner : Owner := ⟨.program ⟨257⟩, ⟨46818⟩⟩
def transferEvent : Nat := 247428
def frameStart : Nat := 247372
def rule : BoundRule := .sum [.predecessor 0 247426 .coefficient, .predecessor 1 247427 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247426 .coefficient)
      LeftBound247411.bound (LeftBound247411.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound247411.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 247427 .coefficient)
      LeftAuthority247424.bound (LeftAuthority247424.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority247424.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound247411.bound, LeftAuthority247424.bound]
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247411.bound, LeftAuthority247424.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound247411.actual selector witness, LeftAuthority247424.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound247428

namespace LeftBound247431
def owner : Owner := ⟨.program ⟨257⟩, ⟨46819⟩⟩
def transferEvent : Nat := 247431
def frameStart : Nat := 247372
def rule : BoundRule := .identity (.predecessor 0 247430 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 247430 .coefficient)
      LeftBound247428.bound (LeftBound247428.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound247428.derived selector witness)

def rawBound : CoeffClass := LeftBound247428.bound
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound247428.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound247428.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound247431

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
