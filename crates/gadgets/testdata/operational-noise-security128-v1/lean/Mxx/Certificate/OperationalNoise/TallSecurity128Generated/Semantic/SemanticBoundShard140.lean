import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard068
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard139

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound26155
def owner : Owner := ⟨.program ⟨257⟩, ⟨70970⟩⟩
def transferEvent : Nat := 26155
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26153 .coefficient) (.predecessor 1 26154 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26153 .coefficient)
      LeftBound26150.bound (LeftBound26150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26150.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26154 .coefficient)
      LeftAuthority17026.bound (LeftAuthority17026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events066.exact17027RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17026.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound26150.bound LeftAuthority17026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26150.bound, LeftAuthority17026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound26150.actual selector witness) * (LeftAuthority17026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26155

namespace LeftBound26156
def owner : Owner := ⟨.program ⟨257⟩, ⟨70970⟩⟩
def transferEvent : Nat := 26156
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩ [⟨.result 17027 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17027 .coefficient)
      LeftAuthority17026.bound (LeftAuthority17026.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨70968⟩⟩) (rawTerms := some (Proof.Events066.exact17027RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17026.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17026.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17026.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17026.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26156

namespace LeftBound26157
def owner : Owner := ⟨.program ⟨257⟩, ⟨70970⟩⟩
def transferEvent : Nat := 26157
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 26152 .summary) (.transfer 26156) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26152 .summary)
      LeftBound26151.bound (LeftBound26151.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69503⟩⟩) (rawTerms := some (Proof.Events102.exact26152RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound26151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 26156)
      LeftBound26156.bound (LeftBound26156.actual selector witness) := by
  exact .transfer (LeftBound26156.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound26151.bound LeftBound26156.bound
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26151.bound, LeftBound26156.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound26151.actual selector witness) * (LeftBound26156.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26157

namespace LeftBound26236
def owner : Owner := ⟨.program ⟨257⟩, ⟨68285⟩⟩
def transferEvent : Nat := 26236
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 26234 .coefficient) (.value (.predecessor 1 26235 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26234 .coefficient)
      LeftAuthority26232.bound (LeftAuthority26232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26232.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26232.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26235 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority26232.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26232.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority26232.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound26236

namespace LeftBound26240
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def transferEvent : Nat := 26240
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 26238 .coefficient) (.predecessor 1 26239 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 26238 .coefficient)
      LeftBound17166.bound (LeftBound17166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17166.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 26239 .coefficient)
      LeftBound26236.bound (LeftBound26236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events102.exact26237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26236.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26236.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17166.bound LeftBound26236.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17166.bound, LeftBound26236.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17166.actual selector witness) * (LeftBound26236.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26240

namespace LeftBound26241
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def transferEvent : Nat := 26241
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨68283⟩⟩]⟩ [⟨.result 26233 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 26233 .coefficient)
      LeftAuthority26232.bound (LeftAuthority26232.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨68283⟩⟩) (rawTerms := some (Proof.Events102.exact26233RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority26232.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority26232.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority26232.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority26232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority26232.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound26241

namespace LeftBound26242
def owner : Owner := ⟨.program ⟨257⟩, ⟨68286⟩⟩
def transferEvent : Nat := 26242
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 17169 .summary) (.transfer 26241) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17169 .summary)
      LeftBound17167.bound (LeftBound17167.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5443⟩⟩) (rawTerms := some (Proof.Events067.exact17169RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound17167.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 26241)
      LeftBound26241.bound (LeftBound26241.actual selector witness) := by
  exact .transfer (LeftBound26241.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound17167.bound LeftBound26241.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17167.bound, LeftBound26241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound17167.actual selector witness) * (LeftBound26241.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound26242

namespace LeftBound27270
def owner : Owner := ⟨.program ⟨257⟩, ⟨18701⟩⟩
def transferEvent : Nat := 27270
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27268 .coefficient, .predecessor 1 27269 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27268 .coefficient)
      LeftAuthority27266.bound (LeftAuthority27266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27266.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27269 .coefficient)
      LeftAuthority27243.bound (LeftAuthority27243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27243.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority27266.bound, LeftAuthority27243.bound]
def bound : CoeffClass := .finite ⟨91, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority27266.bound, LeftAuthority27243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority27266.actual selector witness, LeftAuthority27243.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27270

namespace LeftBound27274
def owner : Owner := ⟨.program ⟨257⟩, ⟨21921⟩⟩
def transferEvent : Nat := 27274
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27272 .coefficient, .predecessor 1 27273 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27272 .coefficient)
      LeftBound27270.bound (LeftBound27270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27271RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27270.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27273 .coefficient)
      LeftAuthority27220.bound (LeftAuthority27220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27270.bound, LeftAuthority27220.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27270.bound, LeftAuthority27220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27270.actual selector witness, LeftAuthority27220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27274

namespace LeftBound27278
def owner : Owner := ⟨.program ⟨257⟩, ⟨31941⟩⟩
def transferEvent : Nat := 27278
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27276 .coefficient, .predecessor 1 27277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27276 .coefficient)
      LeftBound27274.bound (LeftBound27274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27274.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27277 .coefficient)
      LeftAuthority27197.bound (LeftAuthority27197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27274.bound, LeftAuthority27197.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27274.bound, LeftAuthority27197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27274.actual selector witness, LeftAuthority27197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27278

namespace LeftBound27282
def owner : Owner := ⟨.program ⟨257⟩, ⟨50996⟩⟩
def transferEvent : Nat := 27282
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27280 .coefficient, .predecessor 1 27281 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27280 .coefficient)
      LeftBound27278.bound (LeftBound27278.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27279RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27278.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27281 .coefficient)
      LeftAuthority27174.bound (LeftAuthority27174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27278.bound, LeftAuthority27174.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27278.bound, LeftAuthority27174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27278.actual selector witness, LeftAuthority27174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27282

namespace LeftBound27286
def owner : Owner := ⟨.program ⟨257⟩, ⟨53976⟩⟩
def transferEvent : Nat := 27286
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27284 .coefficient, .predecessor 1 27285 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27284 .coefficient)
      LeftBound27282.bound (LeftBound27282.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27283RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27282.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27282.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27285 .coefficient)
      LeftAuthority27151.bound (LeftAuthority27151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27282.bound, LeftAuthority27151.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27282.bound, LeftAuthority27151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27282.actual selector witness, LeftAuthority27151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27286

namespace LeftBound27290
def owner : Owner := ⟨.program ⟨257⟩, ⟨56956⟩⟩
def transferEvent : Nat := 27290
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27288 .coefficient, .predecessor 1 27289 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27288 .coefficient)
      LeftBound27286.bound (LeftBound27286.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27287RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27286.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27286.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27289 .coefficient)
      LeftAuthority27128.bound (LeftAuthority27128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27286.bound, LeftAuthority27128.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27286.bound, LeftAuthority27128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27286.actual selector witness, LeftAuthority27128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27290

namespace LeftBound27294
def owner : Owner := ⟨.program ⟨257⟩, ⟨59936⟩⟩
def transferEvent : Nat := 27294
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27292 .coefficient, .predecessor 1 27293 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27292 .coefficient)
      LeftBound27290.bound (LeftBound27290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27291RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27290.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27293 .coefficient)
      LeftAuthority27105.bound (LeftAuthority27105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27105.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27105.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27290.bound, LeftAuthority27105.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27290.bound, LeftAuthority27105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27290.actual selector witness, LeftAuthority27105.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27294

namespace LeftBound27298
def owner : Owner := ⟨.program ⟨257⟩, ⟨62916⟩⟩
def transferEvent : Nat := 27298
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27296 .coefficient, .predecessor 1 27297 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27296 .coefficient)
      LeftBound27294.bound (LeftBound27294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27294.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27297 .coefficient)
      LeftAuthority27082.bound (LeftAuthority27082.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27083RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27082.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27082.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27294.bound, LeftAuthority27082.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27294.bound, LeftAuthority27082.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27294.actual selector witness, LeftAuthority27082.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27298

namespace LeftBound27302
def owner : Owner := ⟨.program ⟨257⟩, ⟨65994⟩⟩
def transferEvent : Nat := 27302
def frameStart : Nat := 26833
def rule : BoundRule := .sum [.predecessor 0 27300 .coefficient, .predecessor 1 27301 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 27300 .coefficient)
      LeftBound27298.bound (LeftBound27298.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events106.exact27299RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound27298.bound, RecordedBoundRefines] <;> decide)
      (LeftBound27298.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 27301 .coefficient)
      LeftAuthority27059.bound (LeftAuthority27059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events105.exact27060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority27059.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority27059.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound27298.bound, LeftAuthority27059.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound27298.bound, LeftAuthority27059.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound27298.actual selector witness, LeftAuthority27059.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound27302

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
