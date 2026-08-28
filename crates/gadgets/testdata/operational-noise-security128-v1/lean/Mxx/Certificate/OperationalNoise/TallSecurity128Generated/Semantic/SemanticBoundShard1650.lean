import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1595
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1649

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound244103
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def transferEvent : Nat := 244103
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 244101 .coefficient) (.predecessor 1 244102 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244101 .coefficient)
      LeftBound236867.bound (LeftBound236867.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound236867.bound, RecordedBoundRefines] <;> decide)
      (LeftBound236867.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244102 .coefficient)
      LeftBound244099.bound (LeftBound244099.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244099.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244099.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236867.bound LeftBound244099.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236867.bound, LeftBound244099.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236867.actual selector witness) * (LeftBound244099.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244103

namespace LeftBound244104
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def transferEvent : Nat := 244104
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22349⟩⟩]⟩ [⟨.result 244096 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 244096 .coefficient)
      LeftAuthority244095.bound (LeftAuthority244095.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22349⟩⟩) (rawTerms := some (Proof.Events953.exact244096RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244095.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244095.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority244095.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority244095.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound244104

namespace LeftBound244105
def owner : Owner := ⟨.program ⟨257⟩, ⟨22352⟩⟩
def transferEvent : Nat := 244105
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 236870 .summary) (.transfer 244104) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 236870 .summary)
      LeftBound236868.bound (LeftBound236868.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5563⟩⟩) (rawTerms := some (Proof.Events925.exact236870RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound236868.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 244104)
      LeftBound244104.bound (LeftBound244104.actual selector witness) := by
  exact .transfer (LeftBound244104.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound236868.bound LeftBound244104.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound236868.bound, LeftBound244104.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound236868.actual selector witness) * (LeftBound244104.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244105

namespace LeftBound244184
def owner : Owner := ⟨.program ⟨257⟩, ⟨21447⟩⟩
def transferEvent : Nat := 244184
def frameStart : Nat := 244155
def rule : BoundRule := .product (.predecessor 0 244182 .coefficient) (.predecessor 1 244183 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244182 .coefficient)
      LeftAuthority244180.bound (LeftAuthority244180.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244181RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244180.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244180.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244183 .coefficient)
      LeftAuthority244177.bound (LeftAuthority244177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244177.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority244180.bound LeftAuthority244177.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244180.bound, LeftAuthority244177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority244180.actual selector witness) * (LeftAuthority244177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244184

namespace LeftBound244188
def owner : Owner := ⟨.program ⟨257⟩, ⟨21448⟩⟩
def transferEvent : Nat := 244188
def frameStart : Nat := 244155
def rule : BoundRule := .identity (.predecessor 0 244187 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244187 .coefficient)
      LeftBound244184.bound (LeftBound244184.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244184.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244184.derived selector witness)

def rawBound : CoeffClass := LeftBound244184.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound244184.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound244188

namespace LeftBound244205
def owner : Owner := ⟨.program ⟨257⟩, ⟨23198⟩⟩
def transferEvent : Nat := 244205
def frameStart : Nat := 244155
def rule : BoundRule := .sum [.predecessor 0 244203 .coefficient, .predecessor 1 244204 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244203 .coefficient)
      LeftBound244188.bound (LeftBound244188.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound244188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244204 .coefficient)
      LeftAuthority244201.bound (LeftAuthority244201.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority244201.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound244188.bound, LeftAuthority244201.bound]
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244188.bound, LeftAuthority244201.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound244188.actual selector witness, LeftAuthority244201.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound244205

namespace LeftBound244208
def owner : Owner := ⟨.program ⟨257⟩, ⟨23199⟩⟩
def transferEvent : Nat := 244208
def frameStart : Nat := 244155
def rule : BoundRule := .identity (.predecessor 0 244207 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244207 .coefficient)
      LeftBound244205.bound (LeftBound244205.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound244205.derived selector witness)

def rawBound : CoeffClass := LeftBound244205.bound
def bound : CoeffClass := .finite ⟨16, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244205.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound244205.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound244208

namespace LeftBound244214
def owner : Owner := ⟨.program ⟨257⟩, ⟨23200⟩⟩
def transferEvent : Nat := 244214
def frameStart : Nat := 244155
def rule : BoundRule := .product (.predecessor 0 244212 .coefficient) (.predecessor 1 244213 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244212 .coefficient)
      LeftAuthority244210.bound (LeftAuthority244210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244213 .coefficient)
      LeftBound244208.bound (LeftBound244208.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244209RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244208.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244208.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority244210.bound LeftBound244208.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244210.bound, LeftBound244208.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority244210.actual selector witness) * (LeftBound244208.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244214

namespace LeftBound244230
def owner : Owner := ⟨.program ⟨257⟩, ⟨9575⟩⟩
def transferEvent : Nat := 244230
def frameStart : Nat := 244155
def rule : BoundRule := .scale (.predecessor 0 244228 .coefficient) (.value (.predecessor 1 244229 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244228 .coefficient)
      LeftAuthority244226.bound (LeftAuthority244226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244227RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244226.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244229 .coefficient)
      LeftAuthority244217.bound (LeftAuthority244217.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority244217.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority244226.bound LeftAuthority244217.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244226.bound, LeftAuthority244217.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority244226.actual selector witness) * (LeftAuthority244217.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound244230

namespace LeftBound244233
def owner : Owner := ⟨.program ⟨257⟩, ⟨7286⟩⟩
def transferEvent : Nat := 244233
def frameStart : Nat := 244155
def rule : BoundRule := .identity (.predecessor 0 244232 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244232 .coefficient)
      LeftAuthority244220.bound (LeftAuthority244220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244220.derived selector witness)

def rawBound : CoeffClass := LeftAuthority244220.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority244220.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound244233

namespace LeftBound244237
def owner : Owner := ⟨.program ⟨257⟩, ⟨9576⟩⟩
def transferEvent : Nat := 244237
def frameStart : Nat := 244155
def rule : BoundRule := .product (.predecessor 0 244235 .coefficient) (.predecessor 1 244236 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244235 .coefficient)
      LeftBound244233.bound (LeftBound244233.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244233.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244236 .coefficient)
      LeftBound244230.bound (LeftBound244230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244230.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound244233.bound LeftBound244230.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244233.bound, LeftBound244230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound244233.actual selector witness) * (LeftBound244230.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244237

namespace LeftBound244242
def owner : Owner := ⟨.program ⟨257⟩, ⟨23201⟩⟩
def transferEvent : Nat := 244242
def frameStart : Nat := 244155
def rule : BoundRule := .sum [.predecessor 0 244240 .coefficient, .predecessor 1 244241 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244240 .coefficient)
      LeftBound244237.bound (LeftBound244237.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244239RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244237.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244237.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244241 .coefficient)
      LeftBound244214.bound (LeftBound244214.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244216RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244214.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244214.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound244237.bound, LeftBound244214.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244237.bound, LeftBound244214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound244237.actual selector witness, LeftBound244214.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound244242

namespace LeftBound244246
def owner : Owner := ⟨.program ⟨257⟩, ⟨23420⟩⟩
def transferEvent : Nat := 244246
def frameStart : Nat := 244155
def rule : BoundRule := .product (.predecessor 0 244244 .coefficient) (.predecessor 1 244245 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244244 .coefficient)
      LeftBound244242.bound (LeftBound244242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244245 .coefficient)
      LeftAuthority244199.bound (LeftAuthority244199.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244199.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244199.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound244242.bound LeftAuthority244199.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244242.bound, LeftAuthority244199.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound244242.actual selector witness) * (LeftAuthority244199.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244246

namespace LeftBound244257
def owner : Owner := ⟨.program ⟨257⟩, ⟨21794⟩⟩
def transferEvent : Nat := 244257
def frameStart : Nat := 244155
def rule : BoundRule := .product (.predecessor 0 244255 .coefficient) (.predecessor 1 244256 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244255 .coefficient)
      LeftAuthority244210.bound (LeftAuthority244210.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events953.exact244211RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244210.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244210.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244256 .coefficient)
      LeftAuthority244253.bound (LeftAuthority244253.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244253.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244253.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority244210.bound LeftAuthority244253.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244210.bound, LeftAuthority244253.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority244210.actual selector witness) * (LeftAuthority244253.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound244257

namespace LeftBound244265
def owner : Owner := ⟨.program ⟨257⟩, ⟨21795⟩⟩
def transferEvent : Nat := 244265
def frameStart : Nat := 244155
def rule : BoundRule := .sum [.predecessor 0 244263 .coefficient, .predecessor 1 244264 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244263 .coefficient)
      LeftAuthority244261.bound (LeftAuthority244261.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244262RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority244261.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority244261.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244264 .coefficient)
      LeftBound244257.bound (LeftBound244257.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244259RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244257.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244257.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority244261.bound, LeftBound244257.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority244261.bound, LeftBound244257.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority244261.actual selector witness, LeftBound244257.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound244265

namespace LeftBound244269
def owner : Owner := ⟨.program ⟨257⟩, ⟨23421⟩⟩
def transferEvent : Nat := 244269
def frameStart : Nat := 244155
def rule : BoundRule := .sum [.predecessor 0 244267 .coefficient, .predecessor 1 244268 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 244267 .coefficient)
      LeftBound244265.bound (LeftBound244265.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244266RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244265.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244265.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 244268 .coefficient)
      LeftBound244246.bound (LeftBound244246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events954.exact244251RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound244246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound244246.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound244265.bound, LeftBound244246.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound244265.bound, LeftBound244246.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound244265.actual selector witness, LeftBound244246.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound244269

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
