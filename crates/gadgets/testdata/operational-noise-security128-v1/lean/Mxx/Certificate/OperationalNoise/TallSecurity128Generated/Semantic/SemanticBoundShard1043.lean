import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard130
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard982
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard985
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1042

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound156670
def owner : Owner := ⟨.program ⟨257⟩, ⟨23274⟩⟩
def transferEvent : Nat := 156670
def frameStart : Nat := 156614
def rule : BoundRule := .sum [.predecessor 0 156668 .coefficient, .predecessor 1 156669 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156668 .coefficient)
      LeftBound156653.bound (LeftBound156653.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound156653.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156669 .coefficient)
      LeftAuthority156666.bound (LeftAuthority156666.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority156666.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156653.bound, LeftAuthority156666.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156653.bound, LeftAuthority156666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156653.actual selector witness, LeftAuthority156666.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156670

namespace LeftBound156673
def owner : Owner := ⟨.program ⟨257⟩, ⟨23275⟩⟩
def transferEvent : Nat := 156673
def frameStart : Nat := 156614
def rule : BoundRule := .identity (.predecessor 0 156672 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156672 .coefficient)
      LeftBound156670.bound (LeftBound156670.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound156670.derived selector witness)

def rawBound : CoeffClass := LeftBound156670.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156670.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound156670.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound156673

namespace LeftBound156679
def owner : Owner := ⟨.program ⟨257⟩, ⟨23276⟩⟩
def transferEvent : Nat := 156679
def frameStart : Nat := 156614
def rule : BoundRule := .product (.predecessor 0 156677 .coefficient) (.predecessor 1 156678 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156677 .coefficient)
      LeftAuthority156675.bound (LeftAuthority156675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156678 .coefficient)
      LeftBound156673.bound (LeftBound156673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156673.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156673.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority156675.bound LeftBound156673.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority156675.bound, LeftBound156673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority156675.actual selector witness) * (LeftBound156673.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156679

namespace LeftBound156687
def owner : Owner := ⟨.program ⟨257⟩, ⟨23277⟩⟩
def transferEvent : Nat := 156687
def frameStart : Nat := 156614
def rule : BoundRule := .sum [.predecessor 0 156685 .coefficient, .predecessor 1 156686 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156685 .coefficient)
      LeftAuthority156683.bound (LeftAuthority156683.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156684RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156683.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156683.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156686 .coefficient)
      LeftBound156679.bound (LeftBound156679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156681RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156679.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority156683.bound, LeftBound156679.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority156683.bound, LeftBound156679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority156683.actual selector witness, LeftBound156679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156687

namespace LeftBound156691
def owner : Owner := ⟨.program ⟨257⟩, ⟨23780⟩⟩
def transferEvent : Nat := 156691
def frameStart : Nat := 156614
def rule : BoundRule := .product (.predecessor 0 156689 .coefficient) (.predecessor 1 156690 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156689 .coefficient)
      LeftBound156687.bound (LeftBound156687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156687.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156687.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156690 .coefficient)
      LeftAuthority156664.bound (LeftAuthority156664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events611.exact156665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156664.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound156687.bound LeftAuthority156664.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156687.bound, LeftAuthority156664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound156687.actual selector witness) * (LeftAuthority156664.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156691

namespace LeftBound156702
def owner : Owner := ⟨.program ⟨257⟩, ⟨22031⟩⟩
def transferEvent : Nat := 156702
def frameStart : Nat := 156614
def rule : BoundRule := .product (.predecessor 0 156700 .coefficient) (.predecessor 1 156701 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156700 .coefficient)
      LeftAuthority156675.bound (LeftAuthority156675.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156676RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156675.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156675.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156701 .coefficient)
      LeftAuthority156698.bound (LeftAuthority156698.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156699RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156698.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156698.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority156675.bound LeftAuthority156698.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority156675.bound, LeftAuthority156698.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority156675.actual selector witness) * (LeftAuthority156698.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156702

namespace LeftBound156710
def owner : Owner := ⟨.program ⟨257⟩, ⟨22032⟩⟩
def transferEvent : Nat := 156710
def frameStart : Nat := 156614
def rule : BoundRule := .sum [.predecessor 0 156708 .coefficient, .predecessor 1 156709 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156708 .coefficient)
      LeftAuthority156706.bound (LeftAuthority156706.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156707RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority156706.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority156706.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156709 .coefficient)
      LeftBound156702.bound (LeftBound156702.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156704RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156702.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156702.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority156706.bound, LeftBound156702.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority156706.bound, LeftBound156702.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority156706.actual selector witness, LeftBound156702.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156710

namespace LeftBound156714
def owner : Owner := ⟨.program ⟨257⟩, ⟨23784⟩⟩
def transferEvent : Nat := 156714
def frameStart : Nat := 156614
def rule : BoundRule := .sum [.predecessor 0 156712 .coefficient, .predecessor 1 156713 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156712 .coefficient)
      LeftBound156710.bound (LeftBound156710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156710.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156710.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156713 .coefficient)
      LeftBound156691.bound (LeftBound156691.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156696RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156691.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156691.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156710.bound, LeftBound156691.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156710.bound, LeftBound156691.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156710.actual selector witness, LeftBound156691.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156714

namespace LeftBound156727
def owner : Owner := ⟨.program ⟨257⟩, ⟨23782⟩⟩
def transferEvent : Nat := 156727
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 156725 .coefficient, .predecessor 1 156726 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156725 .coefficient)
      LeftBound156556.bound (LeftBound156556.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156724RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156556.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156556.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156726 .coefficient)
      LeftBound156539.bound (LeftBound156539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events611.exact156546RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156539.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156539.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156556.bound, LeftBound156539.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156556.bound, LeftBound156539.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156556.actual selector witness, LeftBound156539.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156727

namespace LeftBound156730
def owner : Owner := ⟨.program ⟨257⟩, ⟨23782⟩⟩
def transferEvent : Nat := 156730
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 156724 .summary, .result 156546 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 156724 .summary)
      LeftBound156558.bound (LeftBound156558.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22619⟩⟩) (rawTerms := some (Proof.Events612.exact156724RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound156558.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 156546 .summary)
      LeftBound156541.bound (LeftBound156541.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23781⟩⟩) (rawTerms := some (Proof.Events611.exact156546RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound156541.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156558.bound, LeftBound156541.bound]
def bound : CoeffClass := .finite ⟨32189003662929394266751515230208, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156558.bound, LeftBound156541.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156558.actual selector witness, LeftBound156541.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156730

namespace LeftBound156754
def owner : Owner := ⟨.program ⟨257⟩, ⟨18205⟩⟩
def transferEvent : Nat := 156754
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 156752 .coefficient) (.predecessor 1 156753 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156752 .coefficient)
      LeftAuthority7193.bound (LeftAuthority7193.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7193.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7193.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156753 .coefficient)
      LeftBound149026.bound (LeftBound149026.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149026.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149026.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority7193.bound LeftBound149026.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7193.bound, LeftBound149026.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority7193.actual selector witness) * (LeftBound149026.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound156754

namespace LeftBound156759
def owner : Owner := ⟨.program ⟨257⟩, ⟨8269⟩⟩
def transferEvent : Nat := 156759
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 156757 .coefficient) (.predecessor 1 156758 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156757 .coefficient)
      LeftBound148897.bound (LeftBound148897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events581.exact148898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound148897.bound, RecordedBoundRefines] <;> decide)
      (LeftBound148897.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156758 .coefficient)
      LeftBound25095.bound (LeftBound25095.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25096RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25095.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25095.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound148897.bound LeftBound25095.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound148897.bound, LeftBound25095.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound148897.actual selector witness) * (LeftBound25095.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156759

namespace LeftBound156764
def owner : Owner := ⟨.program ⟨257⟩, ⟨18206⟩⟩
def transferEvent : Nat := 156764
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 156762 .coefficient, .predecessor 1 156763 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156762 .coefficient)
      LeftBound156759.bound (LeftBound156759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156761RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156763 .coefficient)
      LeftBound156754.bound (LeftBound156754.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156754.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156754.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156759.bound, LeftBound156754.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156759.bound, LeftBound156754.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156759.actual selector witness, LeftBound156754.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156764

namespace LeftBound156768
def owner : Owner := ⟨.program ⟨257⟩, ⟨18207⟩⟩
def transferEvent : Nat := 156768
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 156766 .coefficient, .predecessor 1 156767 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156766 .coefficient)
      LeftBound156764.bound (LeftBound156764.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156765RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156764.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156764.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156767 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound156764.bound, LeftBound25087.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156764.bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound156764.actual selector witness, LeftBound25087.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound156768

namespace LeftBound156769
def owner : Owner := ⟨.program ⟨257⟩, ⟨18207⟩⟩
def transferEvent : Nat := 156769
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨131⟩⟩]⟩ [⟨.result 25088 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 25088 .coefficient)
      LeftBound25087.bound (LeftBound25087.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨131⟩⟩) (rawTerms := some (Proof.Events098.exact25088RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25087.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound25087.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound25087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound25087.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound156769

namespace LeftBound156774
def owner : Owner := ⟨.program ⟨257⟩, ⟨18208⟩⟩
def transferEvent : Nat := 156774
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 156772 .coefficient) (.predecessor 1 156773 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 156772 .coefficient)
      LeftBound156768.bound (LeftBound156768.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events612.exact156771RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound156768.bound, RecordedBoundRefines] <;> decide)
      (LeftBound156768.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 156773 .coefficient)
      LeftAuthority7196.bound (LeftAuthority7196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events028.exact7197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7196.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound156768.bound LeftAuthority7196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound156768.bound, LeftAuthority7196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound156768.actual selector witness) * (LeftAuthority7196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound156774

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
