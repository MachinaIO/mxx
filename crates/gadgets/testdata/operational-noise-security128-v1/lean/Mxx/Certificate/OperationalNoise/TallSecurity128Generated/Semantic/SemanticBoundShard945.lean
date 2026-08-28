import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard134
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard881
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard883
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard885
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard944

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound142414
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def transferEvent : Nat := 142414
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨19316⟩⟩]⟩ [⟨.result 142406 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142406 .coefficient)
      LeftAuthority142405.bound (LeftAuthority142405.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨19316⟩⟩) (rawTerms := some (Proof.Events556.exact142406RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142405.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142405.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority142405.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142405.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority142405.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound142414

namespace LeftBound142415
def owner : Owner := ⟨.program ⟨257⟩, ⟨19319⟩⟩
def transferEvent : Nat := 142415
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 134495 .summary) (.transfer 142414) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 134495 .summary)
      LeftBound134493.bound (LeftBound134493.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5473⟩⟩) (rawTerms := some (Proof.Events525.exact134495RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound134493.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 142414)
      LeftBound142414.bound (LeftBound142414.actual selector witness) := by
  exact .transfer (LeftBound142414.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound134493.bound LeftBound142414.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134493.bound, LeftBound142414.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound134493.actual selector witness) * (LeftBound142414.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142415

namespace LeftBound142510
def owner : Owner := ⟨.program ⟨257⟩, ⟨18533⟩⟩
def transferEvent : Nat := 142510
def frameStart : Nat := 142471
def rule : BoundRule := .identity (.predecessor 0 142509 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142509 .coefficient)
      LeftAuthority142507.bound (LeftAuthority142507.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142508RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142507.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142507.derived selector witness)

def rawBound : CoeffClass := LeftAuthority142507.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority142507.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound142510

namespace LeftBound142527
def owner : Owner := ⟨.program ⟨257⟩, ⟨20038⟩⟩
def transferEvent : Nat := 142527
def frameStart : Nat := 142471
def rule : BoundRule := .sum [.predecessor 0 142525 .coefficient, .predecessor 1 142526 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142525 .coefficient)
      LeftBound142510.bound (LeftBound142510.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound142510.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142526 .coefficient)
      LeftAuthority142523.bound (LeftAuthority142523.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority142523.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142510.bound, LeftAuthority142523.bound]
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142510.bound, LeftAuthority142523.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142510.actual selector witness, LeftAuthority142523.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142527

namespace LeftBound142530
def owner : Owner := ⟨.program ⟨257⟩, ⟨20039⟩⟩
def transferEvent : Nat := 142530
def frameStart : Nat := 142471
def rule : BoundRule := .identity (.predecessor 0 142529 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142529 .coefficient)
      LeftBound142527.bound (LeftBound142527.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound142527.derived selector witness)

def rawBound : CoeffClass := LeftBound142527.bound
def bound : CoeffClass := .finite ⟨3, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142527.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound142527.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound142530

namespace LeftBound142536
def owner : Owner := ⟨.program ⟨257⟩, ⟨20040⟩⟩
def transferEvent : Nat := 142536
def frameStart : Nat := 142471
def rule : BoundRule := .product (.predecessor 0 142534 .coefficient) (.predecessor 1 142535 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142534 .coefficient)
      LeftAuthority142532.bound (LeftAuthority142532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142535 .coefficient)
      LeftBound142530.bound (LeftBound142530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142530.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142530.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority142532.bound LeftBound142530.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142532.bound, LeftBound142530.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority142532.actual selector witness) * (LeftBound142530.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142536

namespace LeftBound142544
def owner : Owner := ⟨.program ⟨257⟩, ⟨20041⟩⟩
def transferEvent : Nat := 142544
def frameStart : Nat := 142471
def rule : BoundRule := .sum [.predecessor 0 142542 .coefficient, .predecessor 1 142543 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142542 .coefficient)
      LeftAuthority142540.bound (LeftAuthority142540.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142541RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142540.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142540.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142543 .coefficient)
      LeftBound142536.bound (LeftBound142536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142536.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142536.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority142540.bound, LeftBound142536.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142540.bound, LeftBound142536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority142540.actual selector witness, LeftBound142536.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142544

namespace LeftBound142548
def owner : Owner := ⟨.program ⟨257⟩, ⟨20436⟩⟩
def transferEvent : Nat := 142548
def frameStart : Nat := 142471
def rule : BoundRule := .product (.predecessor 0 142546 .coefficient) (.predecessor 1 142547 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142546 .coefficient)
      LeftBound142544.bound (LeftBound142544.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142544.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142544.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142547 .coefficient)
      LeftAuthority142521.bound (LeftAuthority142521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142521.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound142544.bound LeftAuthority142521.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142544.bound, LeftAuthority142521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound142544.actual selector witness) * (LeftAuthority142521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142548

namespace LeftBound142559
def owner : Owner := ⟨.program ⟨257⟩, ⟨18735⟩⟩
def transferEvent : Nat := 142559
def frameStart : Nat := 142471
def rule : BoundRule := .product (.predecessor 0 142557 .coefficient) (.predecessor 1 142558 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142557 .coefficient)
      LeftAuthority142532.bound (LeftAuthority142532.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142533RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142532.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142532.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142558 .coefficient)
      LeftAuthority142555.bound (LeftAuthority142555.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142556RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142555.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142555.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority142532.bound LeftAuthority142555.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142532.bound, LeftAuthority142555.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority142532.actual selector witness) * (LeftAuthority142555.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142559

namespace LeftBound142567
def owner : Owner := ⟨.program ⟨257⟩, ⟨18736⟩⟩
def transferEvent : Nat := 142567
def frameStart : Nat := 142471
def rule : BoundRule := .sum [.predecessor 0 142565 .coefficient, .predecessor 1 142566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142565 .coefficient)
      LeftAuthority142563.bound (LeftAuthority142563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority142563.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority142563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142566 .coefficient)
      LeftBound142559.bound (LeftBound142559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142561RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142559.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority142563.bound, LeftBound142559.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority142563.bound, LeftBound142559.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority142563.actual selector witness, LeftBound142559.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142567

namespace LeftBound142571
def owner : Owner := ⟨.program ⟨257⟩, ⟨20440⟩⟩
def transferEvent : Nat := 142571
def frameStart : Nat := 142471
def rule : BoundRule := .sum [.predecessor 0 142569 .coefficient, .predecessor 1 142570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142569 .coefficient)
      LeftBound142567.bound (LeftBound142567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142570 .coefficient)
      LeftBound142548.bound (LeftBound142548.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142548.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142548.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142567.bound, LeftBound142548.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142567.bound, LeftBound142548.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142567.actual selector witness, LeftBound142548.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142571

namespace LeftBound142584
def owner : Owner := ⟨.program ⟨257⟩, ⟨20438⟩⟩
def transferEvent : Nat := 142584
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 142582 .coefficient, .predecessor 1 142583 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142582 .coefficient)
      LeftBound142413.bound (LeftBound142413.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142581RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142413.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142413.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142583 .coefficient)
      LeftBound142396.bound (LeftBound142396.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events556.exact142403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142396.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142396.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142413.bound, LeftBound142396.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142413.bound, LeftBound142396.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142413.actual selector witness, LeftBound142396.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142584

namespace LeftBound142587
def owner : Owner := ⟨.program ⟨257⟩, ⟨20438⟩⟩
def transferEvent : Nat := 142587
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 142581 .summary, .result 142403 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142581 .summary)
      LeftBound142415.bound (LeftBound142415.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨19319⟩⟩) (rawTerms := some (Proof.Events556.exact142581RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142415.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 142403 .summary)
      LeftBound142398.bound (LeftBound142398.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20437⟩⟩) (rawTerms := some (Proof.Events556.exact142403RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound142398.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142415.bound, LeftBound142398.bound]
def bound : CoeffClass := .finite ⟨32188905437706550578131070353408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142415.bound, LeftBound142398.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142415.actual selector witness, LeftBound142398.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142587

namespace LeftBound142611
def owner : Owner := ⟨.program ⟨257⟩, ⟨15309⟩⟩
def transferEvent : Nat := 142611
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 142609 .coefficient) (.predecessor 1 142610 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142609 .coefficient)
      LeftAuthority6468.bound (LeftAuthority6468.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events025.exact6469RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority6468.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority6468.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142610 .coefficient)
      LeftBound134401.bound (LeftBound134401.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events525.exact134403RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134401.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134401.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority6468.bound LeftBound134401.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority6468.bound, LeftBound134401.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority6468.actual selector witness) * (LeftBound134401.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound142611

namespace LeftBound142616
def owner : Owner := ⟨.program ⟨257⟩, ⟨7812⟩⟩
def transferEvent : Nat := 142616
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 142614 .coefficient) (.predecessor 1 142615 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142614 .coefficient)
      LeftBound134272.bound (LeftBound134272.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events524.exact134273RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound134272.bound, RecordedBoundRefines] <;> decide)
      (LeftBound134272.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142615 .coefficient)
      LeftBound25596.bound (LeftBound25596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events099.exact25597RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound25596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound25596.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound134272.bound LeftBound25596.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound134272.bound, LeftBound25596.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound134272.actual selector witness) * (LeftBound25596.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound142616

namespace LeftBound142621
def owner : Owner := ⟨.program ⟨257⟩, ⟨15310⟩⟩
def transferEvent : Nat := 142621
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 142619 .coefficient, .predecessor 1 142620 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 142619 .coefficient)
      LeftBound142616.bound (LeftBound142616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events557.exact142618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142616.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 142620 .coefficient)
      LeftBound142611.bound (LeftBound142611.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events557.exact142613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound142611.bound, RecordedBoundRefines] <;> decide)
      (LeftBound142611.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound142616.bound, LeftBound142611.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound142616.bound, LeftBound142611.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound142616.actual selector witness, LeftBound142611.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound142621

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
