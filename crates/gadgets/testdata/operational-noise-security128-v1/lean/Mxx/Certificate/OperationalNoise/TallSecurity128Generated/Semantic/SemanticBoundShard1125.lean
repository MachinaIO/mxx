import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1124

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound168670
def owner : Owner := ⟨.program ⟨257⟩, ⟨61242⟩⟩
def transferEvent : Nat := 168670
def frameStart : Nat := 168620
def rule : BoundRule := .sum [.predecessor 0 168668 .coefficient, .predecessor 1 168669 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168668 .coefficient)
      LeftBound168653.bound (LeftBound168653.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound168653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168669 .coefficient)
      LeftAuthority168666.bound (LeftAuthority168666.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority168666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound168653.bound, LeftAuthority168666.bound]
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168653.bound, LeftAuthority168666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound168653.actual selector witness, LeftAuthority168666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168670

namespace LeftBound168673
def owner : Owner := ⟨.program ⟨257⟩, ⟨61243⟩⟩
def transferEvent : Nat := 168673
def frameStart : Nat := 168620
def rule : BoundRule := .identity (.predecessor 0 168672 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168672 .coefficient)
      LeftBound168670.bound (LeftBound168670.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound168670.derived selector witness)

def rawBound : CoeffClass := LeftBound168670.bound
def bound : CoeffClass := .finite ⟨324, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound168670.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound168673

namespace LeftBound168679
def owner : Owner := ⟨.program ⟨257⟩, ⟨61244⟩⟩
def transferEvent : Nat := 168679
def frameStart : Nat := 168620
def rule : BoundRule := .product (.predecessor 0 168677 .coefficient) (.predecessor 1 168678 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168677 .coefficient)
      LeftAuthority168675.bound (LeftAuthority168675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168678 .coefficient)
      LeftBound168673.bound (LeftBound168673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168673.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority168675.bound LeftBound168673.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168675.bound, LeftBound168673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority168675.actual selector witness) * (LeftBound168673.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168679

namespace LeftBound168695
def owner : Owner := ⟨.program ⟨257⟩, ⟨9536⟩⟩
def transferEvent : Nat := 168695
def frameStart : Nat := 168620
def rule : BoundRule := .scale (.predecessor 0 168693 .coefficient) (.value (.predecessor 1 168694 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168693 .coefficient)
      LeftAuthority168691.bound (LeftAuthority168691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168692RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168691.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168691.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168694 .coefficient)
      LeftAuthority168682.bound (LeftAuthority168682.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority168682.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority168691.bound LeftAuthority168682.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168691.bound, LeftAuthority168682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority168691.actual selector witness) * (LeftAuthority168682.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound168695

namespace LeftBound168698
def owner : Owner := ⟨.program ⟨257⟩, ⟨7291⟩⟩
def transferEvent : Nat := 168698
def frameStart : Nat := 168620
def rule : BoundRule := .identity (.predecessor 0 168697 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168697 .coefficient)
      LeftAuthority168685.bound (LeftAuthority168685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168685.derived selector witness)

def rawBound : CoeffClass := LeftAuthority168685.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority168685.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound168698

namespace LeftBound168702
def owner : Owner := ⟨.program ⟨257⟩, ⟨9537⟩⟩
def transferEvent : Nat := 168702
def frameStart : Nat := 168620
def rule : BoundRule := .product (.predecessor 0 168700 .coefficient) (.predecessor 1 168701 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168700 .coefficient)
      LeftBound168698.bound (LeftBound168698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168698.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168698.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168701 .coefficient)
      LeftBound168695.bound (LeftBound168695.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168695.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168695.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound168698.bound LeftBound168695.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168698.bound, LeftBound168695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound168698.actual selector witness) * (LeftBound168695.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168702

namespace LeftBound168707
def owner : Owner := ⟨.program ⟨257⟩, ⟨61245⟩⟩
def transferEvent : Nat := 168707
def frameStart : Nat := 168620
def rule : BoundRule := .sum [.predecessor 0 168705 .coefficient, .predecessor 1 168706 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168705 .coefficient)
      LeftBound168702.bound (LeftBound168702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168702.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168706 .coefficient)
      LeftBound168679.bound (LeftBound168679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound168702.bound, LeftBound168679.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168702.bound, LeftBound168679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound168702.actual selector witness, LeftBound168679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168707

namespace LeftBound168711
def owner : Owner := ⟨.program ⟨257⟩, ⟨61506⟩⟩
def transferEvent : Nat := 168711
def frameStart : Nat := 168620
def rule : BoundRule := .product (.predecessor 0 168709 .coefficient) (.predecessor 1 168710 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168709 .coefficient)
      LeftBound168707.bound (LeftBound168707.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168708RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168707.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168707.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168710 .coefficient)
      LeftAuthority168664.bound (LeftAuthority168664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound168707.bound LeftAuthority168664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168707.bound, LeftAuthority168664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound168707.actual selector witness) * (LeftAuthority168664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168711

namespace LeftBound168722
def owner : Owner := ⟨.program ⟨257⟩, ⟨59862⟩⟩
def transferEvent : Nat := 168722
def frameStart : Nat := 168620
def rule : BoundRule := .product (.predecessor 0 168720 .coefficient) (.predecessor 1 168721 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168720 .coefficient)
      LeftAuthority168675.bound (LeftAuthority168675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168721 .coefficient)
      LeftAuthority168718.bound (LeftAuthority168718.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168719RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168718.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168718.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority168675.bound LeftAuthority168718.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168675.bound, LeftAuthority168718.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority168675.actual selector witness) * (LeftAuthority168718.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168722

namespace LeftBound168730
def owner : Owner := ⟨.program ⟨257⟩, ⟨59863⟩⟩
def transferEvent : Nat := 168730
def frameStart : Nat := 168620
def rule : BoundRule := .sum [.predecessor 0 168728 .coefficient, .predecessor 1 168729 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168728 .coefficient)
      LeftAuthority168726.bound (LeftAuthority168726.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168727RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168726.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168726.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168729 .coefficient)
      LeftBound168722.bound (LeftBound168722.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168722.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168722.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority168726.bound, LeftBound168722.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168726.bound, LeftBound168722.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority168726.actual selector witness, LeftBound168722.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168730

namespace LeftBound168734
def owner : Owner := ⟨.program ⟨257⟩, ⟨61507⟩⟩
def transferEvent : Nat := 168734
def frameStart : Nat := 168620
def rule : BoundRule := .sum [.predecessor 0 168732 .coefficient, .predecessor 1 168733 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168732 .coefficient)
      LeftBound168730.bound (LeftBound168730.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168731RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168730.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168730.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168733 .coefficient)
      LeftBound168711.bound (LeftBound168711.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168716RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168711.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168711.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound168730.bound, LeftBound168711.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168730.bound, LeftBound168711.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound168730.actual selector witness, LeftBound168711.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168734

namespace LeftBound168747
def owner : Owner := ⟨.program ⟨257⟩, ⟨61505⟩⟩
def transferEvent : Nat := 168747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 168745 .coefficient, .predecessor 1 168746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168745 .coefficient)
      LeftBound168568.bound (LeftBound168568.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168568.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168568.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168746 .coefficient)
      LeftBound168551.bound (LeftBound168551.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168551.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168551.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound168568.bound, LeftBound168551.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168568.bound, LeftBound168551.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound168568.actual selector witness, LeftBound168551.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168747

namespace LeftBound168750
def owner : Owner := ⟨.program ⟨257⟩, ⟨61505⟩⟩
def transferEvent : Nat := 168750
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 168744 .summary, .result 168558 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168744 .summary)
      LeftBound168570.bound (LeftBound168570.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨60432⟩⟩) (rawTerms := some (Proof.Events659.exact168744RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound168570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168558 .summary)
      LeftBound168553.bound (LeftBound168553.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61504⟩⟩) (rawTerms := some (Proof.Events658.exact168558RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound168553.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound168570.bound, LeftBound168553.bound]
def bound : CoeffClass := .finite ⟨2997962647681031733248, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168570.bound, LeftBound168553.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound168570.actual selector witness, LeftBound168553.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound168750

namespace LeftBound168754
def owner : Owner := ⟨.program ⟨257⟩, ⟨62018⟩⟩
def transferEvent : Nat := 168754
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 168752 .coefficient) (.predecessor 1 168753 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 168752 .coefficient)
      LeftBound168747.bound (LeftBound168747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events659.exact168751RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound168747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound168747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 168753 .coefficient)
      LeftAuthority168473.bound (LeftAuthority168473.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events658.exact168474RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168473.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168473.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound168747.bound LeftAuthority168473.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168747.bound, LeftAuthority168473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound168747.actual selector witness) * (LeftAuthority168473.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168754

namespace LeftBound168755
def owner : Owner := ⟨.program ⟨257⟩, ⟨62018⟩⟩
def transferEvent : Nat := 168755
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨62016⟩⟩]⟩ [⟨.result 168474 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168474 .coefficient)
      LeftAuthority168473.bound (LeftAuthority168473.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨62016⟩⟩) (rawTerms := some (Proof.Events658.exact168474RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority168473.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority168473.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority168473.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority168473.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority168473.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound168755

namespace LeftBound168756
def owner : Owner := ⟨.program ⟨257⟩, ⟨62018⟩⟩
def transferEvent : Nat := 168756
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 168751 .summary) (.transfer 168755) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 168751 .summary)
      LeftBound168750.bound (LeftBound168750.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨61505⟩⟩) (rawTerms := some (Proof.Events659.exact168751RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound168750.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 168755)
      LeftBound168755.bound (LeftBound168755.actual selector witness) := by
  exact .transfer (LeftBound168755.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound168750.bound LeftBound168755.bound
def bound : CoeffClass := .finite ⟨32190378816049003834595889643520, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound168750.bound, LeftBound168755.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound168750.actual selector witness) * (LeftBound168755.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound168756

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
