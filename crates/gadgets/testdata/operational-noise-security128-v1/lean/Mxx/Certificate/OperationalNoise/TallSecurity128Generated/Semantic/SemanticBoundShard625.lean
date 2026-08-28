import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard624

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound96492
def owner : Owner := ⟨.program ⟨257⟩, ⟨53662⟩⟩
def transferEvent : Nat := 96492
def frameStart : Nat := 96459
def rule : BoundRule := .identity (.predecessor 0 96491 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96491 .coefficient)
      LeftBound96488.bound (LeftBound96488.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96490RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96488.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96488.derived selector witness)

def rawBound : CoeffClass := LeftBound96488.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96488.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound96488.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96492

namespace LeftBound96509
def owner : Owner := ⟨.program ⟨257⟩, ⟨55286⟩⟩
def transferEvent : Nat := 96509
def frameStart : Nat := 96459
def rule : BoundRule := .sum [.predecessor 0 96507 .coefficient, .predecessor 1 96508 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96507 .coefficient)
      LeftBound96492.bound (LeftBound96492.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96492.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96508 .coefficient)
      LeftAuthority96505.bound (LeftAuthority96505.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96492.bound, LeftAuthority96505.bound]
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96492.bound, LeftAuthority96505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound96492.actual selector witness, LeftAuthority96505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96509

namespace LeftBound96512
def owner : Owner := ⟨.program ⟨257⟩, ⟨55287⟩⟩
def transferEvent : Nat := 96512
def frameStart : Nat := 96459
def rule : BoundRule := .identity (.predecessor 0 96511 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96511 .coefficient)
      LeftBound96509.bound (LeftBound96509.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound96509.derived selector witness)

def rawBound : CoeffClass := LeftBound96509.bound
def bound : CoeffClass := .finite ⟨144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96509.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound96509.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96512

namespace LeftBound96518
def owner : Owner := ⟨.program ⟨257⟩, ⟨55288⟩⟩
def transferEvent : Nat := 96518
def frameStart : Nat := 96459
def rule : BoundRule := .product (.predecessor 0 96516 .coefficient) (.predecessor 1 96517 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96516 .coefficient)
      LeftAuthority96514.bound (LeftAuthority96514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96517 .coefficient)
      LeftBound96512.bound (LeftBound96512.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96513RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96512.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96512.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority96514.bound LeftBound96512.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96514.bound, LeftBound96512.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority96514.actual selector witness) * (LeftBound96512.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96518

namespace LeftBound96534
def owner : Owner := ⟨.program ⟨257⟩, ⟨9530⟩⟩
def transferEvent : Nat := 96534
def frameStart : Nat := 96459
def rule : BoundRule := .scale (.predecessor 0 96532 .coefficient) (.value (.predecessor 1 96533 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96532 .coefficient)
      LeftAuthority96530.bound (LeftAuthority96530.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96531RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96530.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96530.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96533 .coefficient)
      LeftAuthority96521.bound (LeftAuthority96521.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority96521.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority96530.bound LeftAuthority96521.bound
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96530.bound, LeftAuthority96521.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority96530.actual selector witness) * (LeftAuthority96521.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound96534

namespace LeftBound96537
def owner : Owner := ⟨.program ⟨257⟩, ⟨7289⟩⟩
def transferEvent : Nat := 96537
def frameStart : Nat := 96459
def rule : BoundRule := .identity (.predecessor 0 96536 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96536 .coefficient)
      LeftAuthority96524.bound (LeftAuthority96524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96524.derived selector witness)

def rawBound : CoeffClass := LeftAuthority96524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority96524.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound96537

namespace LeftBound96541
def owner : Owner := ⟨.program ⟨257⟩, ⟨9531⟩⟩
def transferEvent : Nat := 96541
def frameStart : Nat := 96459
def rule : BoundRule := .product (.predecessor 0 96539 .coefficient) (.predecessor 1 96540 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96539 .coefficient)
      LeftBound96537.bound (LeftBound96537.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96538RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96537.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96537.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96540 .coefficient)
      LeftBound96534.bound (LeftBound96534.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96535RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96534.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96534.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound96537.bound LeftBound96534.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96537.bound, LeftBound96534.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound96537.actual selector witness) * (LeftBound96534.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96541

namespace LeftBound96546
def owner : Owner := ⟨.program ⟨257⟩, ⟨55289⟩⟩
def transferEvent : Nat := 96546
def frameStart : Nat := 96459
def rule : BoundRule := .sum [.predecessor 0 96544 .coefficient, .predecessor 1 96545 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96544 .coefficient)
      LeftBound96541.bound (LeftBound96541.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96541.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96541.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96545 .coefficient)
      LeftBound96518.bound (LeftBound96518.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96520RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96518.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96518.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96541.bound, LeftBound96518.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96541.bound, LeftBound96518.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound96541.actual selector witness, LeftBound96518.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96546

namespace LeftBound96550
def owner : Owner := ⟨.program ⟨257⟩, ⟨55557⟩⟩
def transferEvent : Nat := 96550
def frameStart : Nat := 96459
def rule : BoundRule := .product (.predecessor 0 96548 .coefficient) (.predecessor 1 96549 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96548 .coefficient)
      LeftBound96546.bound (LeftBound96546.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96547RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96546.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96546.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96549 .coefficient)
      LeftAuthority96503.bound (LeftAuthority96503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96503.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound96546.bound LeftAuthority96503.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96546.bound, LeftAuthority96503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound96546.actual selector witness) * (LeftAuthority96503.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96550

namespace LeftBound96561
def owner : Owner := ⟨.program ⟨257⟩, ⟨53910⟩⟩
def transferEvent : Nat := 96561
def frameStart : Nat := 96459
def rule : BoundRule := .product (.predecessor 0 96559 .coefficient) (.predecessor 1 96560 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96559 .coefficient)
      LeftAuthority96514.bound (LeftAuthority96514.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96515RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96514.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96514.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96560 .coefficient)
      LeftAuthority96557.bound (LeftAuthority96557.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96558RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96557.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96557.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority96514.bound LeftAuthority96557.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96514.bound, LeftAuthority96557.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority96514.actual selector witness) * (LeftAuthority96557.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96561

namespace LeftBound96569
def owner : Owner := ⟨.program ⟨257⟩, ⟨53911⟩⟩
def transferEvent : Nat := 96569
def frameStart : Nat := 96459
def rule : BoundRule := .sum [.predecessor 0 96567 .coefficient, .predecessor 1 96568 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96567 .coefficient)
      LeftAuthority96565.bound (LeftAuthority96565.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96566RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96565.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96565.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96568 .coefficient)
      LeftBound96561.bound (LeftBound96561.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96561.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96561.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority96565.bound, LeftBound96561.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96565.bound, LeftBound96561.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority96565.actual selector witness, LeftBound96561.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96569

namespace LeftBound96573
def owner : Owner := ⟨.program ⟨257⟩, ⟨55558⟩⟩
def transferEvent : Nat := 96573
def frameStart : Nat := 96459
def rule : BoundRule := .sum [.predecessor 0 96571 .coefficient, .predecessor 1 96572 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96571 .coefficient)
      LeftBound96569.bound (LeftBound96569.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96570RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96569.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96569.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96572 .coefficient)
      LeftBound96550.bound (LeftBound96550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96555RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96569.bound, LeftBound96550.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96569.bound, LeftBound96550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound96569.actual selector witness, LeftBound96550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96573

namespace LeftBound96586
def owner : Owner := ⟨.program ⟨257⟩, ⟨55556⟩⟩
def transferEvent : Nat := 96586
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 96584 .coefficient, .predecessor 1 96585 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96584 .coefficient)
      LeftBound96407.bound (LeftBound96407.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96407.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96407.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96585 .coefficient)
      LeftBound96390.bound (LeftBound96390.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96397RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96390.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96390.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96407.bound, LeftBound96390.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96407.bound, LeftBound96390.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound96407.actual selector witness, LeftBound96390.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96586

namespace LeftBound96589
def owner : Owner := ⟨.program ⟨257⟩, ⟨55556⟩⟩
def transferEvent : Nat := 96589
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 96583 .summary, .result 96397 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96583 .summary)
      LeftBound96409.bound (LeftBound96409.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨54482⟩⟩) (rawTerms := some (Proof.Events377.exact96583RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96409.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96397 .summary)
      LeftBound96392.bound (LeftBound96392.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨55555⟩⟩) (rawTerms := some (Proof.Events376.exact96397RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound96392.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound96409.bound, LeftBound96392.bound]
def bound : CoeffClass := .finite ⟨2997907760060573155328, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96409.bound, LeftBound96392.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound96409.actual selector witness, LeftBound96392.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound96589

namespace LeftBound96593
def owner : Owner := ⟨.program ⟨257⟩, ⟨56089⟩⟩
def transferEvent : Nat := 96593
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 96591 .coefficient) (.predecessor 1 96592 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 96591 .coefficient)
      LeftBound96586.bound (LeftBound96586.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events377.exact96590RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound96586.bound, RecordedBoundRefines] <;> decide)
      (LeftBound96586.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 96592 .coefficient)
      LeftAuthority96312.bound (LeftAuthority96312.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events376.exact96313RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96312.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound96586.bound LeftAuthority96312.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound96586.bound, LeftAuthority96312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound96586.actual selector witness) * (LeftAuthority96312.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound96593

namespace LeftBound96594
def owner : Owner := ⟨.program ⟨257⟩, ⟨56089⟩⟩
def transferEvent : Nat := 96594
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨56087⟩⟩]⟩ [⟨.result 96313 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 96313 .coefficient)
      LeftAuthority96312.bound (LeftAuthority96312.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨56087⟩⟩) (rawTerms := some (Proof.Events376.exact96313RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority96312.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority96312.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority96312.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority96312.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority96312.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound96594

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
