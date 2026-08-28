import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard343

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound56563
def owner : Owner := ⟨.program ⟨257⟩, ⟨67164⟩⟩
def transferEvent : Nat := 56563
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56561 .coefficient, .predecessor 1 56562 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56561 .coefficient)
      LeftBound56559.bound (LeftBound56559.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56559.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56559.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56562 .coefficient)
      LeftAuthority56266.bound (LeftAuthority56266.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56267RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56266.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56266.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56559.bound, LeftAuthority56266.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56559.bound, LeftAuthority56266.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56559.actual selector witness, LeftAuthority56266.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56563

namespace LeftBound56567
def owner : Owner := ⟨.program ⟨257⟩, ⟨67165⟩⟩
def transferEvent : Nat := 56567
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56565 .coefficient, .predecessor 1 56566 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56565 .coefficient)
      LeftBound56563.bound (LeftBound56563.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56564RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56563.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56563.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56566 .coefficient)
      LeftAuthority56243.bound (LeftAuthority56243.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56244RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56243.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56243.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56563.bound, LeftAuthority56243.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56563.bound, LeftAuthority56243.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56563.actual selector witness, LeftAuthority56243.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56567

namespace LeftBound56571
def owner : Owner := ⟨.program ⟨257⟩, ⟨67166⟩⟩
def transferEvent : Nat := 56571
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56569 .coefficient, .predecessor 1 56570 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56569 .coefficient)
      LeftBound56567.bound (LeftBound56567.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56567.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56567.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56570 .coefficient)
      LeftAuthority56220.bound (LeftAuthority56220.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56221RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56220.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56220.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56567.bound, LeftAuthority56220.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56567.bound, LeftAuthority56220.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56567.actual selector witness, LeftAuthority56220.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56571

namespace LeftBound56575
def owner : Owner := ⟨.program ⟨257⟩, ⟨67167⟩⟩
def transferEvent : Nat := 56575
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56573 .coefficient, .predecessor 1 56574 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56573 .coefficient)
      LeftBound56571.bound (LeftBound56571.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events220.exact56572RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56571.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56571.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56574 .coefficient)
      LeftAuthority56197.bound (LeftAuthority56197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56198RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56197.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56197.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56571.bound, LeftAuthority56197.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56571.bound, LeftAuthority56197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56571.actual selector witness, LeftAuthority56197.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56575

namespace LeftBound56579
def owner : Owner := ⟨.program ⟨257⟩, ⟨67168⟩⟩
def transferEvent : Nat := 56579
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56577 .coefficient, .predecessor 1 56578 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56577 .coefficient)
      LeftBound56575.bound (LeftBound56575.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56575.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56575.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56578 .coefficient)
      LeftAuthority56174.bound (LeftAuthority56174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56174.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56174.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56575.bound, LeftAuthority56174.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56575.bound, LeftAuthority56174.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56575.actual selector witness, LeftAuthority56174.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56579

namespace LeftBound56583
def owner : Owner := ⟨.program ⟨257⟩, ⟨67169⟩⟩
def transferEvent : Nat := 56583
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56581 .coefficient, .predecessor 1 56582 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56581 .coefficient)
      LeftBound56579.bound (LeftBound56579.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56580RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56579.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56579.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56582 .coefficient)
      LeftAuthority56151.bound (LeftAuthority56151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56151.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56151.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56579.bound, LeftAuthority56151.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56579.bound, LeftAuthority56151.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56579.actual selector witness, LeftAuthority56151.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56583

namespace LeftBound56587
def owner : Owner := ⟨.program ⟨257⟩, ⟨67170⟩⟩
def transferEvent : Nat := 56587
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56585 .coefficient, .predecessor 1 56586 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56585 .coefficient)
      LeftBound56583.bound (LeftBound56583.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56583.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56583.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56586 .coefficient)
      LeftAuthority56128.bound (LeftAuthority56128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events219.exact56129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56128.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56583.bound, LeftAuthority56128.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56583.bound, LeftAuthority56128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56583.actual selector witness, LeftAuthority56128.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56587

namespace LeftBound56590
def owner : Owner := ⟨.program ⟨257⟩, ⟨67171⟩⟩
def transferEvent : Nat := 56590
def frameStart : Nat := 56086
def rule : BoundRule := .identity (.predecessor 0 56589 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56589 .coefficient)
      LeftBound56587.bound (LeftBound56587.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56588RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56587.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56587.derived selector witness)

def rawBound : CoeffClass := LeftBound56587.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56587.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound56587.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56590

namespace LeftBound56607
def owner : Owner := ⟨.program ⟨257⟩, ⟨69119⟩⟩
def transferEvent : Nat := 56607
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56605 .coefficient, .predecessor 1 56606 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56605 .coefficient)
      LeftBound56590.bound (LeftBound56590.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56590.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56606 .coefficient)
      LeftAuthority56603.bound (LeftAuthority56603.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority56603.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56590.bound, LeftAuthority56603.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56590.bound, LeftAuthority56603.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56590.actual selector witness, LeftAuthority56603.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56607

namespace LeftBound56610
def owner : Owner := ⟨.program ⟨257⟩, ⟨69120⟩⟩
def transferEvent : Nat := 56610
def frameStart : Nat := 56086
def rule : BoundRule := .identity (.predecessor 0 56609 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56609 .coefficient)
      LeftBound56607.bound (LeftBound56607.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound56607.derived selector witness)

def rawBound : CoeffClass := LeftBound56607.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56607.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound56607.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound56610

namespace LeftBound56616
def owner : Owner := ⟨.program ⟨257⟩, ⟨69121⟩⟩
def transferEvent : Nat := 56616
def frameStart : Nat := 56086
def rule : BoundRule := .product (.predecessor 0 56614 .coefficient) (.predecessor 1 56615 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56614 .coefficient)
      LeftAuthority56612.bound (LeftAuthority56612.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56613RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56615 .coefficient)
      LeftBound56610.bound (LeftBound56610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority56612.bound LeftBound56610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56612.bound, LeftBound56610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority56612.actual selector witness) * (LeftBound56610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound56616

namespace LeftBound56692
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 56692
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56690 .coefficient, .predecessor 1 56691 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56690 .coefficient)
      LeftAuthority56688.bound (LeftAuthority56688.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56689RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56688.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56688.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56691 .coefficient)
      LeftAuthority56685.bound (LeftAuthority56685.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56686RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56685.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56685.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority56688.bound, LeftAuthority56685.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority56688.bound, LeftAuthority56685.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority56688.actual selector witness, LeftAuthority56685.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56692

namespace LeftBound56696
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 56696
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56694 .coefficient, .predecessor 1 56695 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56694 .coefficient)
      LeftBound56692.bound (LeftBound56692.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56693RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56692.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56692.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56695 .coefficient)
      LeftAuthority56682.bound (LeftAuthority56682.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56683RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56682.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56692.bound, LeftAuthority56682.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56692.bound, LeftAuthority56682.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56692.actual selector witness, LeftAuthority56682.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56696

namespace LeftBound56700
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 56700
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56698 .coefficient, .predecessor 1 56699 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56698 .coefficient)
      LeftBound56696.bound (LeftBound56696.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56697RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56696.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56696.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56699 .coefficient)
      LeftAuthority56679.bound (LeftAuthority56679.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56680RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56679.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56679.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56696.bound, LeftAuthority56679.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56696.bound, LeftAuthority56679.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56696.actual selector witness, LeftAuthority56679.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56700

namespace LeftBound56704
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 56704
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56702 .coefficient, .predecessor 1 56703 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56702 .coefficient)
      LeftBound56700.bound (LeftBound56700.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56701RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56700.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56700.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56703 .coefficient)
      LeftAuthority56676.bound (LeftAuthority56676.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56677RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56676.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56676.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56700.bound, LeftAuthority56676.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56700.bound, LeftAuthority56676.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56700.actual selector witness, LeftAuthority56676.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56704

namespace LeftBound56708
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 56708
def frameStart : Nat := 56086
def rule : BoundRule := .sum [.predecessor 0 56706 .coefficient, .predecessor 1 56707 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 56706 .coefficient)
      LeftBound56704.bound (LeftBound56704.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56705RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound56704.bound, RecordedBoundRefines] <;> decide)
      (LeftBound56704.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 56707 .coefficient)
      LeftAuthority56673.bound (LeftAuthority56673.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events221.exact56674RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority56673.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority56673.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound56704.bound, LeftAuthority56673.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound56704.bound, LeftAuthority56673.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound56704.actual selector witness, LeftAuthority56673.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound56708

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
