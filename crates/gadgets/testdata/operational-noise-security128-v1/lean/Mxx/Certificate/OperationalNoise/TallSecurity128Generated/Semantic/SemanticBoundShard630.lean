import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard576
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard579
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard629

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound97206
def owner : Owner := ⟨.program ⟨257⟩, ⟨52386⟩⟩
def transferEvent : Nat := 97206
def frameStart : Nat := 97150
def rule : BoundRule := .sum [.predecessor 0 97204 .coefficient, .predecessor 1 97205 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97204 .coefficient)
      LeftBound97189.bound (LeftBound97189.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97205 .coefficient)
      LeftAuthority97202.bound (LeftAuthority97202.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority97202.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97189.bound, LeftAuthority97202.bound]
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97189.bound, LeftAuthority97202.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97189.actual selector witness, LeftAuthority97202.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97206

namespace LeftBound97209
def owner : Owner := ⟨.program ⟨257⟩, ⟨52387⟩⟩
def transferEvent : Nat := 97209
def frameStart : Nat := 97150
def rule : BoundRule := .identity (.predecessor 0 97208 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97208 .coefficient)
      LeftBound97206.bound (LeftBound97206.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound97206.derived selector witness)

def rawBound : CoeffClass := LeftBound97206.bound
def bound : CoeffClass := .finite ⟨10, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97206.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound97206.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound97209

namespace LeftBound97215
def owner : Owner := ⟨.program ⟨257⟩, ⟨52388⟩⟩
def transferEvent : Nat := 97215
def frameStart : Nat := 97150
def rule : BoundRule := .product (.predecessor 0 97213 .coefficient) (.predecessor 1 97214 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97213 .coefficient)
      LeftAuthority97211.bound (LeftAuthority97211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97214 .coefficient)
      LeftBound97209.bound (LeftBound97209.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97209.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97209.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority97211.bound LeftBound97209.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97211.bound, LeftBound97209.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority97211.actual selector witness) * (LeftBound97209.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97215

namespace LeftBound97223
def owner : Owner := ⟨.program ⟨257⟩, ⟨52389⟩⟩
def transferEvent : Nat := 97223
def frameStart : Nat := 97150
def rule : BoundRule := .sum [.predecessor 0 97221 .coefficient, .predecessor 1 97222 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97221 .coefficient)
      LeftAuthority97219.bound (LeftAuthority97219.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97220RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97219.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97219.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97222 .coefficient)
      LeftBound97215.bound (LeftBound97215.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97217RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97215.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97219.bound, LeftBound97215.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97219.bound, LeftBound97215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority97219.actual selector witness, LeftBound97215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97223

namespace LeftBound97227
def owner : Owner := ⟨.program ⟨257⟩, ⟨53108⟩⟩
def transferEvent : Nat := 97227
def frameStart : Nat := 97150
def rule : BoundRule := .product (.predecessor 0 97225 .coefficient) (.predecessor 1 97226 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97225 .coefficient)
      LeftBound97223.bound (LeftBound97223.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97224RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97223.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97223.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97226 .coefficient)
      LeftAuthority97200.bound (LeftAuthority97200.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97201RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97200.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound97223.bound LeftAuthority97200.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97223.bound, LeftAuthority97200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound97223.actual selector witness) * (LeftAuthority97200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97227

namespace LeftBound97238
def owner : Owner := ⟨.program ⟨257⟩, ⟨51258⟩⟩
def transferEvent : Nat := 97238
def frameStart : Nat := 97150
def rule : BoundRule := .product (.predecessor 0 97236 .coefficient) (.predecessor 1 97237 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97236 .coefficient)
      LeftAuthority97211.bound (LeftAuthority97211.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97212RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97211.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97211.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97237 .coefficient)
      LeftAuthority97234.bound (LeftAuthority97234.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97235RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97234.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97234.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority97211.bound LeftAuthority97234.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97211.bound, LeftAuthority97234.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority97211.actual selector witness) * (LeftAuthority97234.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97238

namespace LeftBound97246
def owner : Owner := ⟨.program ⟨257⟩, ⟨51259⟩⟩
def transferEvent : Nat := 97246
def frameStart : Nat := 97150
def rule : BoundRule := .sum [.predecessor 0 97244 .coefficient, .predecessor 1 97245 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97244 .coefficient)
      LeftAuthority97242.bound (LeftAuthority97242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97243RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority97242.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority97242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97245 .coefficient)
      LeftBound97238.bound (LeftBound97238.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97240RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97238.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97238.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority97242.bound, LeftBound97238.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority97242.bound, LeftBound97238.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority97242.actual selector witness, LeftBound97238.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97246

namespace LeftBound97250
def owner : Owner := ⟨.program ⟨257⟩, ⟨53112⟩⟩
def transferEvent : Nat := 97250
def frameStart : Nat := 97150
def rule : BoundRule := .sum [.predecessor 0 97248 .coefficient, .predecessor 1 97249 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97248 .coefficient)
      LeftBound97246.bound (LeftBound97246.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97247RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97246.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97246.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97249 .coefficient)
      LeftBound97227.bound (LeftBound97227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97246.bound, LeftBound97227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97246.bound, LeftBound97227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97246.actual selector witness, LeftBound97227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97250

namespace LeftBound97263
def owner : Owner := ⟨.program ⟨257⟩, ⟨53110⟩⟩
def transferEvent : Nat := 97263
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97261 .coefficient, .predecessor 1 97262 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97261 .coefficient)
      LeftBound97092.bound (LeftBound97092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97260RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97092.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97092.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97262 .coefficient)
      LeftBound97075.bound (LeftBound97075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events379.exact97082RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97075.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97092.bound, LeftBound97075.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97092.bound, LeftBound97075.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97092.actual selector witness, LeftBound97075.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97263

namespace LeftBound97266
def owner : Owner := ⟨.program ⟨257⟩, ⟨53110⟩⟩
def transferEvent : Nat := 97266
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 97260 .summary, .result 97082 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 97260 .summary)
      LeftBound97094.bound (LeftBound97094.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨51859⟩⟩) (rawTerms := some (Proof.Events379.exact97260RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97094.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 97082 .summary)
      LeftBound97077.bound (LeftBound97077.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨53109⟩⟩) (rawTerms := some (Proof.Events379.exact97082RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound97077.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97094.bound, LeftBound97077.bound]
def bound : CoeffClass := .finite ⟨32189593014266456398474184491008, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97094.bound, LeftBound97077.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97094.actual selector witness, LeftBound97077.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97266

namespace LeftBound97290
def owner : Owner := ⟨.program ⟨257⟩, ⟨24351⟩⟩
def transferEvent : Nat := 97290
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 97288 .coefficient) (.predecessor 1 97289 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97288 .coefficient)
      LeftAuthority4155.bound (LeftAuthority4155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4155.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4155.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97289 .coefficient)
      LeftBound90526.bound (LeftBound90526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90528RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90526.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90526.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority4155.bound LeftBound90526.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority4155.bound, LeftBound90526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority4155.actual selector witness) * (LeftBound90526.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound97290

namespace LeftBound97295
def owner : Owner := ⟨.program ⟨257⟩, ⟨9941⟩⟩
def transferEvent : Nat := 97295
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97293 .coefficient) (.predecessor 1 97294 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97293 .coefficient)
      LeftBound90397.bound (LeftBound90397.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events353.exact90398RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound90397.bound, RecordedBoundRefines] <;> decide)
      (LeftBound90397.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97294 .coefficient)
      LeftBound24093.bound (LeftBound24093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24093.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound90397.bound LeftBound24093.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound90397.bound, LeftBound24093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound90397.actual selector witness) * (LeftBound24093.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97295

namespace LeftBound97300
def owner : Owner := ⟨.program ⟨257⟩, ⟨24352⟩⟩
def transferEvent : Nat := 97300
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97298 .coefficient, .predecessor 1 97299 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97298 .coefficient)
      LeftBound97295.bound (LeftBound97295.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97297RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97295.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97295.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97299 .coefficient)
      LeftBound97290.bound (LeftBound97290.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97292RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97290.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97290.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97295.bound, LeftBound97290.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97295.bound, LeftBound97290.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97295.actual selector witness, LeftBound97290.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97300

namespace LeftBound97304
def owner : Owner := ⟨.program ⟨257⟩, ⟨24353⟩⟩
def transferEvent : Nat := 97304
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 97302 .coefficient, .predecessor 1 97303 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97302 .coefficient)
      LeftBound97300.bound (LeftBound97300.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97301RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97300.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97300.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97303 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound97300.bound, LeftBound24085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97300.bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound97300.actual selector witness, LeftBound24085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound97304

namespace LeftBound97305
def owner : Owner := ⟨.program ⟨257⟩, ⟨24353⟩⟩
def transferEvent : Nat := 97305
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩ [⟨.result 24086 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24086 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨133⟩⟩) (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24085.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24085.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound97305

namespace LeftBound97310
def owner : Owner := ⟨.program ⟨257⟩, ⟨31623⟩⟩
def transferEvent : Nat := 97310
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 97308 .coefficient) (.predecessor 1 97309 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 97308 .coefficient)
      LeftBound97304.bound (LeftBound97304.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events380.exact97307RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound97304.bound, RecordedBoundRefines] <;> decide)
      (LeftBound97304.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 97309 .coefficient)
      LeftAuthority4158.bound (LeftAuthority4158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events016.exact4159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority4158.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority4158.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound97304.bound LeftAuthority4158.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound97304.bound, LeftAuthority4158.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound97304.actual selector witness) * (LeftAuthority4158.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound97310

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
