import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1392
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1415
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1469

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound219092
def owner : Owner := ⟨.program ⟨257⟩, ⟨36627⟩⟩
def transferEvent : Nat := 219092
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 219087 .summary) (.transfer 219091) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219087 .summary)
      LeftBound219086.bound (LeftBound219086.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨36626⟩⟩) (rawTerms := some (Proof.Events855.exact219087RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound219086.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 219091)
      LeftBound219091.bound (LeftBound219091.actual selector witness) := by
  exact .transfer (LeftBound219091.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound219086.bound LeftBound219091.bound
def bound : CoeffClass := .finite ⟨345664763728542925759002774434880600145920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219086.bound, LeftBound219091.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound219086.actual selector witness) * (LeftBound219091.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219092

namespace LeftBound219107
def owner : Owner := ⟨.program ⟨257⟩, ⟨30965⟩⟩
def transferEvent : Nat := 219107
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 219105 .coefficient) (.predecessor 1 219106 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219105 .coefficient)
      LeftBound210694.bound (LeftBound210694.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events823.exact210698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound210694.bound, RecordedBoundRefines] <;> decide)
      (LeftBound210694.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219106 .coefficient)
      LeftAuthority219103.bound (LeftAuthority219103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events855.exact219104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219103.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound210694.bound LeftAuthority219103.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound210694.bound, LeftAuthority219103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound210694.actual selector witness) * (LeftAuthority219103.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219107

namespace LeftBound219108
def owner : Owner := ⟨.program ⟨257⟩, ⟨30965⟩⟩
def transferEvent : Nat := 219108
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨30963⟩⟩]⟩ [⟨.result 219104 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219104 .coefficient)
      LeftAuthority219103.bound (LeftAuthority219103.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨30963⟩⟩) (rawTerms := some (Proof.Events855.exact219104RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219103.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority219103.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority219103.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound219108

namespace LeftBound219109
def owner : Owner := ⟨.program ⟨257⟩, ⟨30965⟩⟩
def transferEvent : Nat := 219109
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 210698 .summary) (.transfer 219108) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 210698 .summary)
      LeftBound210697.bound (LeftBound210697.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨30601⟩⟩) (rawTerms := some (Proof.Events823.exact210698RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound210697.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 219108)
      LeftBound219108.bound (LeftBound219108.actual selector witness) := by
  exact .transfer (LeftBound219108.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound210697.bound LeftBound219108.bound
def bound : CoeffClass := .finite ⟨32192146870060190229763897425920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound210697.bound, LeftBound219108.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound210697.actual selector witness) * (LeftBound219108.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219109

namespace LeftBound219120
def owner : Owner := ⟨.program ⟨257⟩, ⟨29834⟩⟩
def transferEvent : Nat := 219120
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 219118 .coefficient) (.value (.predecessor 1 219119 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219118 .coefficient)
      LeftAuthority219116.bound (LeftAuthority219116.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events855.exact219117RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219116.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219119 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority219116.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219116.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority219116.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound219120

namespace LeftBound219124
def owner : Owner := ⟨.program ⟨257⟩, ⟨29835⟩⟩
def transferEvent : Nat := 219124
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 219122 .coefficient) (.predecessor 1 219123 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219122 .coefficient)
      LeftBound207617.bound (LeftBound207617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound207617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound207617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219123 .coefficient)
      LeftBound219120.bound (LeftBound219120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events855.exact219121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219120.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207617.bound LeftBound219120.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207617.bound, LeftBound219120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207617.actual selector witness) * (LeftBound219120.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219124

namespace LeftBound219125
def owner : Owner := ⟨.program ⟨257⟩, ⟨29835⟩⟩
def transferEvent : Nat := 219125
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨29832⟩⟩]⟩ [⟨.result 219117 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 219117 .coefficient)
      LeftAuthority219116.bound (LeftAuthority219116.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29832⟩⟩) (rawTerms := some (Proof.Events855.exact219117RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219116.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219116.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority219116.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219116.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority219116.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound219125

namespace LeftBound219126
def owner : Owner := ⟨.program ⟨257⟩, ⟨29835⟩⟩
def transferEvent : Nat := 219126
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 207620 .summary) (.transfer 219125) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 207620 .summary)
      LeftBound207618.bound (LeftBound207618.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5599⟩⟩) (rawTerms := some (Proof.Events811.exact207620RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound207618.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 219125)
      LeftBound219125.bound (LeftBound219125.actual selector witness) := by
  exact .transfer (LeftBound219125.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound207618.bound LeftBound219125.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound207618.bound, LeftBound219125.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound207618.actual selector witness) * (LeftBound219125.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219126

namespace LeftBound219221
def owner : Owner := ⟨.program ⟨257⟩, ⟨29089⟩⟩
def transferEvent : Nat := 219221
def frameStart : Nat := 219182
def rule : BoundRule := .identity (.predecessor 0 219220 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219220 .coefficient)
      LeftAuthority219218.bound (LeftAuthority219218.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219219RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219218.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219218.derived selector witness)

def rawBound : CoeffClass := LeftAuthority219218.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219218.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority219218.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound219221

namespace LeftBound219238
def owner : Owner := ⟨.program ⟨257⟩, ⟨30446⟩⟩
def transferEvent : Nat := 219238
def frameStart : Nat := 219182
def rule : BoundRule := .sum [.predecessor 0 219236 .coefficient, .predecessor 1 219237 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219236 .coefficient)
      LeftBound219221.bound (LeftBound219221.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound219221.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219237 .coefficient)
      LeftAuthority219234.bound (LeftAuthority219234.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority219234.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound219221.bound, LeftAuthority219234.bound]
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219221.bound, LeftAuthority219234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound219221.actual selector witness, LeftAuthority219234.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219238

namespace LeftBound219241
def owner : Owner := ⟨.program ⟨257⟩, ⟨30447⟩⟩
def transferEvent : Nat := 219241
def frameStart : Nat := 219182
def rule : BoundRule := .identity (.predecessor 0 219240 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219240 .coefficient)
      LeftBound219238.bound (LeftBound219238.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound219238.derived selector witness)

def rawBound : CoeffClass := LeftBound219238.bound
def bound : CoeffClass := .finite ⟨36, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound219238.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound219241

namespace LeftBound219247
def owner : Owner := ⟨.program ⟨257⟩, ⟨30448⟩⟩
def transferEvent : Nat := 219247
def frameStart : Nat := 219182
def rule : BoundRule := .product (.predecessor 0 219245 .coefficient) (.predecessor 1 219246 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219245 .coefficient)
      LeftAuthority219243.bound (LeftAuthority219243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219246 .coefficient)
      LeftBound219241.bound (LeftBound219241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219242RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219241.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority219243.bound LeftBound219241.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219243.bound, LeftBound219241.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority219243.actual selector witness) * (LeftBound219241.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219247

namespace LeftBound219255
def owner : Owner := ⟨.program ⟨257⟩, ⟨30449⟩⟩
def transferEvent : Nat := 219255
def frameStart : Nat := 219182
def rule : BoundRule := .sum [.predecessor 0 219253 .coefficient, .predecessor 1 219254 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219253 .coefficient)
      LeftAuthority219251.bound (LeftAuthority219251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219252RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219251.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219254 .coefficient)
      LeftBound219247.bound (LeftBound219247.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219249RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219247.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219247.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority219251.bound, LeftBound219247.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219251.bound, LeftBound219247.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority219251.actual selector witness, LeftBound219247.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219255

namespace LeftBound219259
def owner : Owner := ⟨.program ⟨257⟩, ⟨30964⟩⟩
def transferEvent : Nat := 219259
def frameStart : Nat := 219182
def rule : BoundRule := .product (.predecessor 0 219257 .coefficient) (.predecessor 1 219258 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219257 .coefficient)
      LeftBound219255.bound (LeftBound219255.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219256RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219255.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219255.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219258 .coefficient)
      LeftAuthority219232.bound (LeftAuthority219232.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219233RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219232.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219232.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound219255.bound LeftAuthority219232.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound219255.bound, LeftAuthority219232.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound219255.actual selector witness) * (LeftAuthority219232.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219259

namespace LeftBound219270
def owner : Owner := ⟨.program ⟨257⟩, ⟨29304⟩⟩
def transferEvent : Nat := 219270
def frameStart : Nat := 219182
def rule : BoundRule := .product (.predecessor 0 219268 .coefficient) (.predecessor 1 219269 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219268 .coefficient)
      LeftAuthority219243.bound (LeftAuthority219243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219269 .coefficient)
      LeftAuthority219266.bound (LeftAuthority219266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219266.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority219243.bound LeftAuthority219266.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219243.bound, LeftAuthority219266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority219243.actual selector witness) * (LeftAuthority219266.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound219270

namespace LeftBound219278
def owner : Owner := ⟨.program ⟨257⟩, ⟨29305⟩⟩
def transferEvent : Nat := 219278
def frameStart : Nat := 219182
def rule : BoundRule := .sum [.predecessor 0 219276 .coefficient, .predecessor 1 219277 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 219276 .coefficient)
      LeftAuthority219274.bound (LeftAuthority219274.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219275RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority219274.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority219274.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 219277 .coefficient)
      LeftBound219270.bound (LeftBound219270.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events856.exact219272RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound219270.bound, RecordedBoundRefines] <;> decide)
      (LeftBound219270.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority219274.bound, LeftBound219270.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority219274.bound, LeftBound219270.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority219274.actual selector witness, LeftBound219270.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound219278

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
