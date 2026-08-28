import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard581
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard640
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard670

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound104155
def owner : Owner := ⟨.program ⟨257⟩, ⟨23308⟩⟩
def transferEvent : Nat := 104155
def frameStart : Nat := 104090
def rule : BoundRule := .product (.predecessor 0 104153 .coefficient) (.predecessor 1 104154 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104153 .coefficient)
      LeftAuthority104151.bound (LeftAuthority104151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104154 .coefficient)
      LeftBound104149.bound (LeftBound104149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104149.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority104151.bound LeftBound104149.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104151.bound, LeftBound104149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority104151.actual selector witness) * (LeftBound104149.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104155

namespace LeftBound104163
def owner : Owner := ⟨.program ⟨257⟩, ⟨23309⟩⟩
def transferEvent : Nat := 104163
def frameStart : Nat := 104090
def rule : BoundRule := .sum [.predecessor 0 104161 .coefficient, .predecessor 1 104162 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104161 .coefficient)
      LeftAuthority104159.bound (LeftAuthority104159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104159.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104162 .coefficient)
      LeftBound104155.bound (LeftBound104155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104157RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104159.bound, LeftBound104155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104159.bound, LeftBound104155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority104159.actual selector witness, LeftBound104155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104163

namespace LeftBound104167
def owner : Owner := ⟨.program ⟨257⟩, ⟨24021⟩⟩
def transferEvent : Nat := 104167
def frameStart : Nat := 104090
def rule : BoundRule := .product (.predecessor 0 104165 .coefficient) (.predecessor 1 104166 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104165 .coefficient)
      LeftBound104163.bound (LeftBound104163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104164RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104163.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104166 .coefficient)
      LeftAuthority104140.bound (LeftAuthority104140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104140.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104140.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound104163.bound LeftAuthority104140.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104163.bound, LeftAuthority104140.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound104163.actual selector witness) * (LeftAuthority104140.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104167

namespace LeftBound104178
def owner : Owner := ⟨.program ⟨257⟩, ⟨22179⟩⟩
def transferEvent : Nat := 104178
def frameStart : Nat := 104090
def rule : BoundRule := .product (.predecessor 0 104176 .coefficient) (.predecessor 1 104177 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104176 .coefficient)
      LeftAuthority104151.bound (LeftAuthority104151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104177 .coefficient)
      LeftAuthority104174.bound (LeftAuthority104174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104174.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority104151.bound LeftAuthority104174.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104151.bound, LeftAuthority104174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority104151.actual selector witness) * (LeftAuthority104174.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104178

namespace LeftBound104186
def owner : Owner := ⟨.program ⟨257⟩, ⟨22180⟩⟩
def transferEvent : Nat := 104186
def frameStart : Nat := 104090
def rule : BoundRule := .sum [.predecessor 0 104184 .coefficient, .predecessor 1 104185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104184 .coefficient)
      LeftAuthority104182.bound (LeftAuthority104182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104182.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104185 .coefficient)
      LeftBound104178.bound (LeftBound104178.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104178.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority104182.bound, LeftBound104178.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104182.bound, LeftBound104178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority104182.actual selector witness, LeftBound104178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104186

namespace LeftBound104190
def owner : Owner := ⟨.program ⟨257⟩, ⟨24026⟩⟩
def transferEvent : Nat := 104190
def frameStart : Nat := 104090
def rule : BoundRule := .sum [.predecessor 0 104188 .coefficient, .predecessor 1 104189 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104188 .coefficient)
      LeftBound104186.bound (LeftBound104186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104189 .coefficient)
      LeftBound104167.bound (LeftBound104167.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104167.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104167.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104186.bound, LeftBound104167.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104186.bound, LeftBound104167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104186.actual selector witness, LeftBound104167.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104190

namespace LeftBound104203
def owner : Owner := ⟨.program ⟨257⟩, ⟨24023⟩⟩
def transferEvent : Nat := 104203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 104201 .coefficient, .predecessor 1 104202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104201 .coefficient)
      LeftBound104032.bound (LeftBound104032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104202 .coefficient)
      LeftBound104015.bound (LeftBound104015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events406.exact104022RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104015.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104032.bound, LeftBound104015.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104032.bound, LeftBound104015.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104032.actual selector witness, LeftBound104015.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104203

namespace LeftBound104206
def owner : Owner := ⟨.program ⟨257⟩, ⟨24023⟩⟩
def transferEvent : Nat := 104206
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 104200 .summary, .result 104022 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104200 .summary)
      LeftBound104034.bound (LeftBound104034.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22775⟩⟩) (rawTerms := some (Proof.Events407.exact104200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104034.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104022 .summary)
      LeftBound104017.bound (LeftBound104017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24022⟩⟩) (rawTerms := some (Proof.Events406.exact104022RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104017.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound104034.bound, LeftBound104017.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104034.bound, LeftBound104017.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound104034.actual selector witness, LeftBound104017.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound104206

namespace LeftBound104210
def owner : Owner := ⟨.program ⟨257⟩, ⟨24024⟩⟩
def transferEvent : Nat := 104210
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104208 .coefficient) (.predecessor 1 104209 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104208 .coefficient)
      LeftBound104203.bound (LeftBound104203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104207RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104209 .coefficient)
      LeftBound15841.bound (LeftBound15841.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15842RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15841.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15841.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound104203.bound LeftBound15841.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104203.bound, LeftBound15841.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound104203.actual selector witness) * (LeftBound15841.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104210

namespace LeftBound104211
def owner : Owner := ⟨.program ⟨257⟩, ⟨24024⟩⟩
def transferEvent : Nat := 104211
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7155⟩⟩]⟩ [⟨.result 15838 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15838 .coefficient)
      LeftAuthority15837.bound (LeftAuthority15837.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7155⟩⟩) (rawTerms := some (Proof.Events061.exact15838RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15837.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15837.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15837.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15837.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15837.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104211

namespace LeftBound104212
def owner : Owner := ⟨.program ⟨257⟩, ⟨24024⟩⟩
def transferEvent : Nat := 104212
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 104207 .summary) (.transfer 104211) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104207 .summary)
      LeftBound104206.bound (LeftBound104206.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24023⟩⟩) (rawTerms := some (Proof.Events407.exact104207RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound104206.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 104211)
      LeftBound104211.bound (LeftBound104211.actual selector witness) := by
  exact .transfer (LeftBound104211.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound104206.bound LeftBound104211.bound
def bound : CoeffClass := .finite ⟨345626795057764889831969145180473178193920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound104206.bound, LeftBound104211.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound104206.actual selector witness) * (LeftBound104211.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104212

namespace LeftBound104227
def owner : Owner := ⟨.program ⟨257⟩, ⟨20802⟩⟩
def transferEvent : Nat := 104227
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104225 .coefficient) (.predecessor 1 104226 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104225 .coefficient)
      LeftBound98514.bound (LeftBound98514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events384.exact98518RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound98514.bound, RecordedBoundRefines] <;> decide)
      (LeftBound98514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104226 .coefficient)
      LeftAuthority104223.bound (LeftAuthority104223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104223.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound98514.bound LeftAuthority104223.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98514.bound, LeftAuthority104223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound98514.actual selector witness) * (LeftAuthority104223.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104227

namespace LeftBound104228
def owner : Owner := ⟨.program ⟨257⟩, ⟨20802⟩⟩
def transferEvent : Nat := 104228
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨20800⟩⟩]⟩ [⟨.result 104224 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 104224 .coefficient)
      LeftAuthority104223.bound (LeftAuthority104223.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨20800⟩⟩) (rawTerms := some (Proof.Events407.exact104224RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104223.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104223.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority104223.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104223.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority104223.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound104228

namespace LeftBound104229
def owner : Owner := ⟨.program ⟨257⟩, ⟨20802⟩⟩
def transferEvent : Nat := 104229
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 98518 .summary) (.transfer 104228) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 98518 .summary)
      LeftBound98517.bound (LeftBound98517.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20276⟩⟩) (rawTerms := some (Proof.Events384.exact98518RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound98517.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 104228)
      LeftBound104228.bound (LeftBound104228.actual selector witness) := by
  exact .transfer (LeftBound104228.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound98517.bound LeftBound104228.bound
def bound : CoeffClass := .finite ⟨32188905437706348505289216491520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound98517.bound, LeftBound104228.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound98517.actual selector witness) * (LeftBound104228.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104229

namespace LeftBound104240
def owner : Owner := ⟨.program ⟨257⟩, ⟨19554⟩⟩
def transferEvent : Nat := 104240
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 104238 .coefficient) (.value (.predecessor 1 104239 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104238 .coefficient)
      LeftAuthority104236.bound (LeftAuthority104236.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104237RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority104236.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority104236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104239 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority104236.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority104236.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority104236.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound104240

namespace LeftBound104244
def owner : Owner := ⟨.program ⟨257⟩, ⟨19555⟩⟩
def transferEvent : Nat := 104244
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 104242 .coefficient) (.predecessor 1 104243 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 104242 .coefficient)
      LeftBound90617.bound (LeftBound90617.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90620RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90617.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90617.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 104243 .coefficient)
      LeftBound104240.bound (LeftBound104240.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events407.exact104241RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound104240.bound, RecordedBoundRefines] <;> decide)
      (LeftBound104240.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound90617.bound LeftBound104240.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90617.bound, LeftBound104240.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound90617.actual selector witness) * (LeftBound104240.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound104244

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
