import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard682
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard733

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound112126
def owner : Owner := ⟨.program ⟨257⟩, ⟨7287⟩⟩
def transferEvent : Nat := 112126
def frameStart : Nat := 112048
def rule : BoundRule := .identity (.predecessor 0 112125 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112125 .coefficient)
      LeftAuthority112113.bound (LeftAuthority112113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112113.derived selector witness)

def rawBound : CoeffClass := LeftAuthority112113.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority112113.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound112126

namespace LeftBound112130
def owner : Owner := ⟨.program ⟨257⟩, ⟨9579⟩⟩
def transferEvent : Nat := 112130
def frameStart : Nat := 112048
def rule : BoundRule := .product (.predecessor 0 112128 .coefficient) (.predecessor 1 112129 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112128 .coefficient)
      LeftBound112126.bound (LeftBound112126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112126.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112129 .coefficient)
      LeftBound112123.bound (LeftBound112123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound112126.bound LeftBound112123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112126.bound, LeftBound112123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound112126.actual selector witness) * (LeftBound112123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112130

namespace LeftBound112135
def owner : Owner := ⟨.program ⟨257⟩, ⟨33233⟩⟩
def transferEvent : Nat := 112135
def frameStart : Nat := 112048
def rule : BoundRule := .sum [.predecessor 0 112133 .coefficient, .predecessor 1 112134 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112133 .coefficient)
      LeftBound112130.bound (LeftBound112130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112132RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112130.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112130.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112134 .coefficient)
      LeftBound112107.bound (LeftBound112107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112109RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112107.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112107.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound112130.bound, LeftBound112107.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112130.bound, LeftBound112107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound112130.actual selector witness, LeftBound112107.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound112135

namespace LeftBound112139
def owner : Owner := ⟨.program ⟨257⟩, ⟨33473⟩⟩
def transferEvent : Nat := 112139
def frameStart : Nat := 112048
def rule : BoundRule := .product (.predecessor 0 112137 .coefficient) (.predecessor 1 112138 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112137 .coefficient)
      LeftBound112135.bound (LeftBound112135.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112135.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112135.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112138 .coefficient)
      LeftAuthority112092.bound (LeftAuthority112092.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112093RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112092.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112092.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound112135.bound LeftAuthority112092.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112135.bound, LeftAuthority112092.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound112135.actual selector witness) * (LeftAuthority112092.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112139

namespace LeftBound112150
def owner : Owner := ⟨.program ⟨257⟩, ⟨31838⟩⟩
def transferEvent : Nat := 112150
def frameStart : Nat := 112048
def rule : BoundRule := .product (.predecessor 0 112148 .coefficient) (.predecessor 1 112149 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112148 .coefficient)
      LeftAuthority112103.bound (LeftAuthority112103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact112104RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112103.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112149 .coefficient)
      LeftAuthority112146.bound (LeftAuthority112146.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112147RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112146.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112146.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority112103.bound LeftAuthority112146.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112103.bound, LeftAuthority112146.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority112103.actual selector witness) * (LeftAuthority112146.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112150

namespace LeftBound112158
def owner : Owner := ⟨.program ⟨257⟩, ⟨31839⟩⟩
def transferEvent : Nat := 112158
def frameStart : Nat := 112048
def rule : BoundRule := .sum [.predecessor 0 112156 .coefficient, .predecessor 1 112157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112156 .coefficient)
      LeftAuthority112154.bound (LeftAuthority112154.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112154.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112154.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112157 .coefficient)
      LeftBound112150.bound (LeftBound112150.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112152RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112150.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112150.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority112154.bound, LeftBound112150.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112154.bound, LeftBound112150.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority112154.actual selector witness, LeftBound112150.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound112158

namespace LeftBound112162
def owner : Owner := ⟨.program ⟨257⟩, ⟨33474⟩⟩
def transferEvent : Nat := 112162
def frameStart : Nat := 112048
def rule : BoundRule := .sum [.predecessor 0 112160 .coefficient, .predecessor 1 112161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112160 .coefficient)
      LeftBound112158.bound (LeftBound112158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112161 .coefficient)
      LeftBound112139.bound (LeftBound112139.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112144RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112139.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112139.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound112158.bound, LeftBound112139.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112158.bound, LeftBound112139.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound112158.actual selector witness, LeftBound112139.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound112162

namespace LeftBound112175
def owner : Owner := ⟨.program ⟨257⟩, ⟨33472⟩⟩
def transferEvent : Nat := 112175
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 112173 .coefficient, .predecessor 1 112174 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112173 .coefficient)
      LeftBound111996.bound (LeftBound111996.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112172RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111996.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111996.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112174 .coefficient)
      LeftBound111979.bound (LeftBound111979.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact111986RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound111979.bound, RecordedBoundRefines] <;> decide)
      (LeftBound111979.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111996.bound, LeftBound111979.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111996.bound, LeftBound111979.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111996.actual selector witness, LeftBound111979.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound112175

namespace LeftBound112178
def owner : Owner := ⟨.program ⟨257⟩, ⟨33472⟩⟩
def transferEvent : Nat := 112178
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 112172 .summary, .result 111986 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112172 .summary)
      LeftBound111998.bound (LeftBound111998.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32402⟩⟩) (rawTerms := some (Proof.Events438.exact112172RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111986 .summary)
      LeftBound111981.bound (LeftBound111981.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33471⟩⟩) (rawTerms := some (Proof.Events437.exact111986RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound111981.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound111998.bound, LeftBound111981.bound]
def bound : CoeffClass := .finite ⟨2997852872440114577408, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound111998.bound, LeftBound111981.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound111998.actual selector witness, LeftBound111981.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound112178

namespace LeftBound112182
def owner : Owner := ⟨.program ⟨257⟩, ⟨33925⟩⟩
def transferEvent : Nat := 112182
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 112180 .coefficient) (.predecessor 1 112181 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112180 .coefficient)
      LeftBound112175.bound (LeftBound112175.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112175.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112181 .coefficient)
      LeftAuthority111901.bound (LeftAuthority111901.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events437.exact111902RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111901.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound112175.bound LeftAuthority111901.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112175.bound, LeftAuthority111901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound112175.actual selector witness) * (LeftAuthority111901.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112182

namespace LeftBound112183
def owner : Owner := ⟨.program ⟨257⟩, ⟨33925⟩⟩
def transferEvent : Nat := 112183
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨33923⟩⟩]⟩ [⟨.result 111902 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 111902 .coefficient)
      LeftAuthority111901.bound (LeftAuthority111901.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨33923⟩⟩) (rawTerms := some (Proof.Events437.exact111902RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority111901.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority111901.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority111901.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority111901.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority111901.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound112183

namespace LeftBound112184
def owner : Owner := ⟨.program ⟨257⟩, ⟨33925⟩⟩
def transferEvent : Nat := 112184
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 112179 .summary) (.transfer 112183) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112179 .summary)
      LeftBound112178.bound (LeftBound112178.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33472⟩⟩) (rawTerms := some (Proof.Events438.exact112179RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound112178.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 112183)
      LeftBound112183.bound (LeftBound112183.actual selector witness) := by
  exact .transfer (LeftBound112183.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound112178.bound LeftBound112183.bound
def bound : CoeffClass := .finite ⟨32189200113374879571150551121920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound112178.bound, LeftBound112183.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound112178.actual selector witness) * (LeftBound112183.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112184

namespace LeftBound112195
def owner : Owner := ⟨.program ⟨257⟩, ⟨32718⟩⟩
def transferEvent : Nat := 112195
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 112193 .coefficient) (.value (.predecessor 1 112194 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112193 .coefficient)
      LeftAuthority112191.bound (LeftAuthority112191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112192RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112194 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority112191.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112191.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority112191.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound112195

namespace LeftBound112199
def owner : Owner := ⟨.program ⟨257⟩, ⟨32719⟩⟩
def transferEvent : Nat := 112199
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 112197 .coefficient) (.predecessor 1 112198 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 112197 .coefficient)
      LeftBound105242.bound (LeftBound105242.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound105242.bound, RecordedBoundRefines] <;> decide)
      (LeftBound105242.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 112198 .coefficient)
      LeftBound112195.bound (LeftBound112195.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events438.exact112196RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound112195.bound, RecordedBoundRefines] <;> decide)
      (LeftBound112195.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105242.bound LeftBound112195.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105242.bound, LeftBound112195.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105242.actual selector witness) * (LeftBound112195.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112199

namespace LeftBound112200
def owner : Owner := ⟨.program ⟨257⟩, ⟨32719⟩⟩
def transferEvent : Nat := 112200
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨32716⟩⟩]⟩ [⟨.result 112192 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 112192 .coefficient)
      LeftAuthority112191.bound (LeftAuthority112191.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32716⟩⟩) (rawTerms := some (Proof.Events438.exact112192RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority112191.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority112191.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority112191.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority112191.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority112191.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound112200

namespace LeftBound112201
def owner : Owner := ⟨.program ⟨257⟩, ⟨32719⟩⟩
def transferEvent : Nat := 112201
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 105245 .summary) (.transfer 112200) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 105245 .summary)
      LeftBound105243.bound (LeftBound105243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨5770⟩⟩) (rawTerms := some (Proof.Events411.exact105245RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound105243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 112200)
      LeftBound112200.bound (LeftBound112200.actual selector witness) := by
  exact .transfer (LeftBound112200.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound105243.bound LeftBound112200.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound105243.bound, LeftBound112200.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound105243.actual selector witness) * (LeftBound112200.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound112201

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
