import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1445
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1449
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1452
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1455

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound216134
def owner : Owner := ⟨.program ⟨257⟩, ⟨17206⟩⟩
def transferEvent : Nat := 216134
def frameStart : Nat := 216078
def rule : BoundRule := .sum [.predecessor 0 216132 .coefficient, .predecessor 1 216133 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216132 .coefficient)
      LeftBound216117.bound (LeftBound216117.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound216117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216133 .coefficient)
      LeftAuthority216130.bound (LeftAuthority216130.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority216130.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216117.bound, LeftAuthority216130.bound]
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216117.bound, LeftAuthority216130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216117.actual selector witness, LeftAuthority216130.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216134

namespace LeftBound216137
def owner : Owner := ⟨.program ⟨257⟩, ⟨17207⟩⟩
def transferEvent : Nat := 216137
def frameStart : Nat := 216078
def rule : BoundRule := .identity (.predecessor 0 216136 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216136 .coefficient)
      LeftBound216134.bound (LeftBound216134.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound216134.derived selector witness)

def rawBound : CoeffClass := LeftBound216134.bound
def bound : CoeffClass := .finite ⟨2, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound216134.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound216137

namespace LeftBound216143
def owner : Owner := ⟨.program ⟨257⟩, ⟨17208⟩⟩
def transferEvent : Nat := 216143
def frameStart : Nat := 216078
def rule : BoundRule := .product (.predecessor 0 216141 .coefficient) (.predecessor 1 216142 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216141 .coefficient)
      LeftAuthority216139.bound (LeftAuthority216139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216142 .coefficient)
      LeftBound216137.bound (LeftBound216137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216137.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority216139.bound LeftBound216137.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216139.bound, LeftBound216137.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority216139.actual selector witness) * (LeftBound216137.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216143

namespace LeftBound216151
def owner : Owner := ⟨.program ⟨257⟩, ⟨17209⟩⟩
def transferEvent : Nat := 216151
def frameStart : Nat := 216078
def rule : BoundRule := .sum [.predecessor 0 216149 .coefficient, .predecessor 1 216150 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216149 .coefficient)
      LeftAuthority216147.bound (LeftAuthority216147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216147.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216150 .coefficient)
      LeftBound216143.bound (LeftBound216143.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216145RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216143.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216143.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority216147.bound, LeftBound216143.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216147.bound, LeftBound216143.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority216147.actual selector witness, LeftBound216143.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216151

namespace LeftBound216155
def owner : Owner := ⟨.program ⟨257⟩, ⟨17762⟩⟩
def transferEvent : Nat := 216155
def frameStart : Nat := 216078
def rule : BoundRule := .product (.predecessor 0 216153 .coefficient) (.predecessor 1 216154 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216153 .coefficient)
      LeftBound216151.bound (LeftBound216151.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216151.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216151.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216154 .coefficient)
      LeftAuthority216128.bound (LeftAuthority216128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound216151.bound LeftAuthority216128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216151.bound, LeftAuthority216128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound216151.actual selector witness) * (LeftAuthority216128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216155

namespace LeftBound216166
def owner : Owner := ⟨.program ⟨257⟩, ⟨16036⟩⟩
def transferEvent : Nat := 216166
def frameStart : Nat := 216078
def rule : BoundRule := .product (.predecessor 0 216164 .coefficient) (.predecessor 1 216165 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216164 .coefficient)
      LeftAuthority216139.bound (LeftAuthority216139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216140RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216139.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216139.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216165 .coefficient)
      LeftAuthority216162.bound (LeftAuthority216162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216162.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216162.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority216139.bound LeftAuthority216162.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216139.bound, LeftAuthority216162.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority216139.actual selector witness) * (LeftAuthority216162.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound216166

namespace LeftBound216174
def owner : Owner := ⟨.program ⟨257⟩, ⟨16037⟩⟩
def transferEvent : Nat := 216174
def frameStart : Nat := 216078
def rule : BoundRule := .sum [.predecessor 0 216172 .coefficient, .predecessor 1 216173 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216172 .coefficient)
      LeftAuthority216170.bound (LeftAuthority216170.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216171RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority216170.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority216170.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216173 .coefficient)
      LeftBound216166.bound (LeftBound216166.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216168RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216166.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216166.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority216170.bound, LeftBound216166.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority216170.bound, LeftBound216166.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority216170.actual selector witness, LeftBound216166.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216174

namespace LeftBound216178
def owner : Owner := ⟨.program ⟨257⟩, ⟨17765⟩⟩
def transferEvent : Nat := 216178
def frameStart : Nat := 216078
def rule : BoundRule := .sum [.predecessor 0 216176 .coefficient, .predecessor 1 216177 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216176 .coefficient)
      LeftBound216174.bound (LeftBound216174.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216175RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216174.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216174.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216177 .coefficient)
      LeftBound216155.bound (LeftBound216155.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216160RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216155.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216155.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216174.bound, LeftBound216155.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216174.bound, LeftBound216155.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216174.actual selector witness, LeftBound216155.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216178

namespace LeftBound216191
def owner : Owner := ⟨.program ⟨257⟩, ⟨17764⟩⟩
def transferEvent : Nat := 216191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216189 .coefficient, .predecessor 1 216190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216189 .coefficient)
      LeftBound216020.bound (LeftBound216020.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216020.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216020.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216190 .coefficient)
      LeftBound216003.bound (LeftBound216003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events843.exact216010RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216003.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216020.bound, LeftBound216003.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216020.bound, LeftBound216003.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216020.actual selector witness, LeftBound216003.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216191

namespace LeftBound216194
def owner : Owner := ⟨.program ⟨257⟩, ⟨17764⟩⟩
def transferEvent : Nat := 216194
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216188 .summary, .result 216010 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216188 .summary)
      LeftBound216022.bound (LeftBound216022.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨16599⟩⟩) (rawTerms := some (Proof.Events844.exact216188RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216022.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216010 .summary)
      LeftBound216005.bound (LeftBound216005.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17763⟩⟩) (rawTerms := some (Proof.Events843.exact216010RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216005.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216022.bound, LeftBound216005.bound]
def bound : CoeffClass := .finite ⟨32188807212483706889510625476608, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216022.bound, LeftBound216005.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216022.actual selector witness, LeftBound216005.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216194

namespace LeftBound216198
def owner : Owner := ⟨.program ⟨257⟩, ⟨20656⟩⟩
def transferEvent : Nat := 216198
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216196 .coefficient, .predecessor 1 216197 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216196 .coefficient)
      LeftBound216191.bound (LeftBound216191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216195RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216197 .coefficient)
      LeftBound215709.bound (LeftBound215709.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events842.exact215713RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215709.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215709.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216191.bound, LeftBound215709.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216191.bound, LeftBound215709.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216191.actual selector witness, LeftBound215709.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216198

namespace LeftBound216199
def owner : Owner := ⟨.program ⟨257⟩, ⟨20656⟩⟩
def transferEvent : Nat := 216199
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216195 .summary, .result 215713 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216195 .summary)
      LeftBound216194.bound (LeftBound216194.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨17764⟩⟩) (rawTerms := some (Proof.Events844.exact216195RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216194.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 215713 .summary)
      LeftBound215712.bound (LeftBound215712.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20655⟩⟩) (rawTerms := some (Proof.Events842.exact215713RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound215712.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216194.bound, LeftBound215712.bound]
def bound : CoeffClass := .finite ⟨64377712650190257467641695830016, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216194.bound, LeftBound215712.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216194.actual selector witness, LeftBound215712.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216199

namespace LeftBound216203
def owner : Owner := ⟨.program ⟨257⟩, ⟨23876⟩⟩
def transferEvent : Nat := 216203
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216201 .coefficient, .predecessor 1 216202 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216201 .coefficient)
      LeftBound216198.bound (LeftBound216198.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216200RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216198.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216198.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216202 .coefficient)
      LeftBound215227.bound (LeftBound215227.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events840.exact215231RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215227.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215227.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216198.bound, LeftBound215227.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216198.bound, LeftBound215227.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216198.actual selector witness, LeftBound215227.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216203

namespace LeftBound216204
def owner : Owner := ⟨.program ⟨257⟩, ⟨23876⟩⟩
def transferEvent : Nat := 216204
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216200 .summary, .result 215231 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216200 .summary)
      LeftBound216199.bound (LeftBound216199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨20656⟩⟩) (rawTerms := some (Proof.Events844.exact216200RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 215231 .summary)
      LeftBound215230.bound (LeftBound215230.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23875⟩⟩) (rawTerms := some (Proof.Events840.exact215231RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound215230.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216199.bound, LeftBound215230.bound]
def bound : CoeffClass := .finite ⟨96566716313119651734393211060224, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216199.bound, LeftBound215230.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216199.actual selector witness, LeftBound215230.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216204

namespace LeftBound216208
def owner : Owner := ⟨.program ⟨257⟩, ⟨33896⟩⟩
def transferEvent : Nat := 216208
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 216206 .coefficient, .predecessor 1 216207 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 216206 .coefficient)
      LeftBound216203.bound (LeftBound216203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events844.exact216205RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound216203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound216203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 216207 .coefficient)
      LeftBound214745.bound (LeftBound214745.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events838.exact214749RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound214745.bound, RecordedBoundRefines] <;> decide)
      (LeftBound214745.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216203.bound, LeftBound214745.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216203.bound, LeftBound214745.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216203.actual selector witness, LeftBound214745.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216208

namespace LeftBound216209
def owner : Owner := ⟨.program ⟨257⟩, ⟨33896⟩⟩
def transferEvent : Nat := 216209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 216205 .summary, .result 214749 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 216205 .summary)
      LeftBound216204.bound (LeftBound216204.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23876⟩⟩) (rawTerms := some (Proof.Events844.exact216205RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound216204.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 214749 .summary)
      LeftBound214748.bound (LeftBound214748.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33895⟩⟩) (rawTerms := some (Proof.Events838.exact214749RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound214748.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound216204.bound, LeftBound214748.bound]
def bound : CoeffClass := .finite ⟨128755916426494733378385616044032, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound216204.bound, LeftBound214748.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound216204.actual selector witness, LeftBound214748.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound216209

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
