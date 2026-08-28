import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard892

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound135543
def owner : Owner := ⟨.program ⟨257⟩, ⟨42307⟩⟩
def transferEvent : Nat := 135543
def frameStart : Nat := 135514
def rule : BoundRule := .product (.predecessor 0 135541 .coefficient) (.predecessor 1 135542 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135541 .coefficient)
      LeftAuthority135539.bound (LeftAuthority135539.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135540RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135539.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135539.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135542 .coefficient)
      LeftAuthority135536.bound (LeftAuthority135536.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135537RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135536.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135536.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority135539.bound LeftAuthority135536.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135539.bound, LeftAuthority135536.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority135539.actual selector witness) * (LeftAuthority135536.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135543

namespace LeftBound135547
def owner : Owner := ⟨.program ⟨257⟩, ⟨42308⟩⟩
def transferEvent : Nat := 135547
def frameStart : Nat := 135514
def rule : BoundRule := .identity (.predecessor 0 135546 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135546 .coefficient)
      LeftBound135543.bound (LeftBound135543.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135545RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135543.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135543.derived selector witness)

def rawBound : CoeffClass := LeftBound135543.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135543.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound135543.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound135547

namespace LeftBound135564
def owner : Owner := ⟨.program ⟨257⟩, ⟨44038⟩⟩
def transferEvent : Nat := 135564
def frameStart : Nat := 135514
def rule : BoundRule := .sum [.predecessor 0 135562 .coefficient, .predecessor 1 135563 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135562 .coefficient)
      LeftBound135547.bound (LeftBound135547.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound135547.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135563 .coefficient)
      LeftAuthority135560.bound (LeftAuthority135560.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority135560.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound135547.bound, LeftAuthority135560.bound]
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135547.bound, LeftAuthority135560.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound135547.actual selector witness, LeftAuthority135560.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135564

namespace LeftBound135567
def owner : Owner := ⟨.program ⟨257⟩, ⟨44039⟩⟩
def transferEvent : Nat := 135567
def frameStart : Nat := 135514
def rule : BoundRule := .identity (.predecessor 0 135566 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135566 .coefficient)
      LeftBound135564.bound (LeftBound135564.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound135564.derived selector witness)

def rawBound : CoeffClass := LeftBound135564.bound
def bound : CoeffClass := .finite ⟨2704, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135564.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound135564.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound135567

namespace LeftBound135573
def owner : Owner := ⟨.program ⟨257⟩, ⟨44040⟩⟩
def transferEvent : Nat := 135573
def frameStart : Nat := 135514
def rule : BoundRule := .product (.predecessor 0 135571 .coefficient) (.predecessor 1 135572 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135571 .coefficient)
      LeftAuthority135569.bound (LeftAuthority135569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135572 .coefficient)
      LeftBound135567.bound (LeftBound135567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135567.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority135569.bound LeftBound135567.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135569.bound, LeftBound135567.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority135569.actual selector witness) * (LeftBound135567.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135573

namespace LeftBound135589
def owner : Owner := ⟨.program ⟨257⟩, ⟨9560⟩⟩
def transferEvent : Nat := 135589
def frameStart : Nat := 135514
def rule : BoundRule := .scale (.predecessor 0 135587 .coefficient) (.value (.predecessor 1 135588 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135587 .coefficient)
      LeftAuthority135585.bound (LeftAuthority135585.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135586RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135585.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135585.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135588 .coefficient)
      LeftAuthority135576.bound (LeftAuthority135576.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority135576.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority135585.bound LeftAuthority135576.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135585.bound, LeftAuthority135576.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority135585.actual selector witness) * (LeftAuthority135576.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound135589

namespace LeftBound135592
def owner : Owner := ⟨.program ⟨257⟩, ⟨7300⟩⟩
def transferEvent : Nat := 135592
def frameStart : Nat := 135514
def rule : BoundRule := .identity (.predecessor 0 135591 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135591 .coefficient)
      LeftAuthority135579.bound (LeftAuthority135579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135579.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135579.derived selector witness)

def rawBound : CoeffClass := LeftAuthority135579.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135579.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority135579.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound135592

namespace LeftBound135596
def owner : Owner := ⟨.program ⟨257⟩, ⟨9561⟩⟩
def transferEvent : Nat := 135596
def frameStart : Nat := 135514
def rule : BoundRule := .product (.predecessor 0 135594 .coefficient) (.predecessor 1 135595 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135594 .coefficient)
      LeftBound135592.bound (LeftBound135592.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135593RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135592.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135595 .coefficient)
      LeftBound135589.bound (LeftBound135589.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135589.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135589.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound135592.bound LeftBound135589.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135592.bound, LeftBound135589.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound135592.actual selector witness) * (LeftBound135589.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135596

namespace LeftBound135601
def owner : Owner := ⟨.program ⟨257⟩, ⟨44041⟩⟩
def transferEvent : Nat := 135601
def frameStart : Nat := 135514
def rule : BoundRule := .sum [.predecessor 0 135599 .coefficient, .predecessor 1 135600 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135599 .coefficient)
      LeftBound135596.bound (LeftBound135596.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135598RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135596.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135596.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135600 .coefficient)
      LeftBound135573.bound (LeftBound135573.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135575RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135573.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135573.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound135596.bound, LeftBound135573.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135596.bound, LeftBound135573.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound135596.actual selector witness, LeftBound135573.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135601

namespace LeftBound135605
def owner : Owner := ⟨.program ⟨257⟩, ⟨44225⟩⟩
def transferEvent : Nat := 135605
def frameStart : Nat := 135514
def rule : BoundRule := .product (.predecessor 0 135603 .coefficient) (.predecessor 1 135604 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135603 .coefficient)
      LeftBound135601.bound (LeftBound135601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135601.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135604 .coefficient)
      LeftAuthority135558.bound (LeftAuthority135558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135559RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135558.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135558.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound135601.bound LeftAuthority135558.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135601.bound, LeftAuthority135558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound135601.actual selector witness) * (LeftAuthority135558.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135605

namespace LeftBound135616
def owner : Owner := ⟨.program ⟨257⟩, ⟨42734⟩⟩
def transferEvent : Nat := 135616
def frameStart : Nat := 135514
def rule : BoundRule := .product (.predecessor 0 135614 .coefficient) (.predecessor 1 135615 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135614 .coefficient)
      LeftAuthority135569.bound (LeftAuthority135569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135569.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135615 .coefficient)
      LeftAuthority135612.bound (LeftAuthority135612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135612.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority135569.bound LeftAuthority135612.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135569.bound, LeftAuthority135612.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority135569.actual selector witness) * (LeftAuthority135612.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135616

namespace LeftBound135624
def owner : Owner := ⟨.program ⟨257⟩, ⟨42735⟩⟩
def transferEvent : Nat := 135624
def frameStart : Nat := 135514
def rule : BoundRule := .sum [.predecessor 0 135622 .coefficient, .predecessor 1 135623 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135622 .coefficient)
      LeftAuthority135620.bound (LeftAuthority135620.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135621RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135620.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135620.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135623 .coefficient)
      LeftBound135616.bound (LeftBound135616.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135618RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135616.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135616.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority135620.bound, LeftBound135616.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority135620.bound, LeftBound135616.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority135620.actual selector witness, LeftBound135616.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135624

namespace LeftBound135628
def owner : Owner := ⟨.program ⟨257⟩, ⟨44226⟩⟩
def transferEvent : Nat := 135628
def frameStart : Nat := 135514
def rule : BoundRule := .sum [.predecessor 0 135626 .coefficient, .predecessor 1 135627 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135626 .coefficient)
      LeftBound135624.bound (LeftBound135624.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135625RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135624.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135624.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135627 .coefficient)
      LeftBound135605.bound (LeftBound135605.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135610RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135605.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135605.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound135624.bound, LeftBound135605.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135624.bound, LeftBound135605.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound135624.actual selector witness, LeftBound135605.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135628

namespace LeftBound135641
def owner : Owner := ⟨.program ⟨257⟩, ⟨44224⟩⟩
def transferEvent : Nat := 135641
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 135639 .coefficient, .predecessor 1 135640 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135639 .coefficient)
      LeftBound135462.bound (LeftBound135462.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135638RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135462.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135462.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135640 .coefficient)
      LeftBound135445.bound (LeftBound135445.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135452RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135445.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135445.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound135462.bound, LeftBound135445.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135462.bound, LeftBound135445.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound135462.actual selector witness, LeftBound135445.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135641

namespace LeftBound135644
def owner : Owner := ⟨.program ⟨257⟩, ⟨44224⟩⟩
def transferEvent : Nat := 135644
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 135638 .summary, .result 135452 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 135638 .summary)
      LeftBound135464.bound (LeftBound135464.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨43162⟩⟩) (rawTerms := some (Proof.Events529.exact135638RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound135464.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 135452 .summary)
      LeftBound135447.bound (LeftBound135447.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨44223⟩⟩) (rawTerms := some (Proof.Events529.exact135452RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound135447.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound135464.bound, LeftBound135447.bound]
def bound : CoeffClass := .finite ⟨2998273677530297008128, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135464.bound, LeftBound135447.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound135464.actual selector witness, LeftBound135447.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound135644

namespace LeftBound135648
def owner : Owner := ⟨.program ⟨257⟩, ⟨44496⟩⟩
def transferEvent : Nat := 135648
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 135646 .coefficient) (.predecessor 1 135647 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 135646 .coefficient)
      LeftBound135641.bound (LeftBound135641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events529.exact135645RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound135641.bound, RecordedBoundRefines] <;> decide)
      (LeftBound135641.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 135647 .coefficient)
      LeftAuthority135367.bound (LeftAuthority135367.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events528.exact135368RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority135367.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority135367.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound135641.bound LeftAuthority135367.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound135641.bound, LeftAuthority135367.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound135641.actual selector witness) * (LeftAuthority135367.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound135648

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
