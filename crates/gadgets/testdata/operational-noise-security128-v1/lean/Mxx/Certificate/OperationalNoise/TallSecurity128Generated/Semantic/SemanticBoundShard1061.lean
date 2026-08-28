import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard051
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard986
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1002
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1060

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound160111
def owner : Owner := ⟨.program ⟨257⟩, ⟨41456⟩⟩
def transferEvent : Nat := 160111
def frameStart : Nat := 160046
def rule : BoundRule := .product (.predecessor 0 160109 .coefficient) (.predecessor 1 160110 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160109 .coefficient)
      LeftAuthority160107.bound (LeftAuthority160107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160107.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160110 .coefficient)
      LeftBound160105.bound (LeftBound160105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160105.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority160107.bound LeftBound160105.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160107.bound, LeftBound160105.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority160107.actual selector witness) * (LeftBound160105.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160111

namespace LeftBound160119
def owner : Owner := ⟨.program ⟨257⟩, ⟨41457⟩⟩
def transferEvent : Nat := 160119
def frameStart : Nat := 160046
def rule : BoundRule := .sum [.predecessor 0 160117 .coefficient, .predecessor 1 160118 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160117 .coefficient)
      LeftAuthority160115.bound (LeftAuthority160115.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160116RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160115.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160115.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160118 .coefficient)
      LeftBound160111.bound (LeftBound160111.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160111.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160111.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority160115.bound, LeftBound160111.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160115.bound, LeftBound160111.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority160115.actual selector witness, LeftBound160111.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound160119

namespace LeftBound160123
def owner : Owner := ⟨.program ⟨257⟩, ⟨41909⟩⟩
def transferEvent : Nat := 160123
def frameStart : Nat := 160046
def rule : BoundRule := .product (.predecessor 0 160121 .coefficient) (.predecessor 1 160122 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160121 .coefficient)
      LeftBound160119.bound (LeftBound160119.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160119.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160119.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160122 .coefficient)
      LeftAuthority160096.bound (LeftAuthority160096.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160097RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160096.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160096.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound160119.bound LeftAuthority160096.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound160119.bound, LeftAuthority160096.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound160119.actual selector witness) * (LeftAuthority160096.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160123

namespace LeftBound160134
def owner : Owner := ⟨.program ⟨257⟩, ⟨40285⟩⟩
def transferEvent : Nat := 160134
def frameStart : Nat := 160046
def rule : BoundRule := .product (.predecessor 0 160132 .coefficient) (.predecessor 1 160133 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160132 .coefficient)
      LeftAuthority160107.bound (LeftAuthority160107.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160108RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160107.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160107.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160133 .coefficient)
      LeftAuthority160130.bound (LeftAuthority160130.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160131RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160130.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160130.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority160107.bound LeftAuthority160130.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160107.bound, LeftAuthority160130.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority160107.actual selector witness) * (LeftAuthority160130.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160134

namespace LeftBound160142
def owner : Owner := ⟨.program ⟨257⟩, ⟨40286⟩⟩
def transferEvent : Nat := 160142
def frameStart : Nat := 160046
def rule : BoundRule := .sum [.predecessor 0 160140 .coefficient, .predecessor 1 160141 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160140 .coefficient)
      LeftAuthority160138.bound (LeftAuthority160138.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160139RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160138.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160138.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160141 .coefficient)
      LeftBound160134.bound (LeftBound160134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160136RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160134.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority160138.bound, LeftBound160134.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160138.bound, LeftBound160134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority160138.actual selector witness, LeftBound160134.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound160142

namespace LeftBound160146
def owner : Owner := ⟨.program ⟨257⟩, ⟨41913⟩⟩
def transferEvent : Nat := 160146
def frameStart : Nat := 160046
def rule : BoundRule := .sum [.predecessor 0 160144 .coefficient, .predecessor 1 160145 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160144 .coefficient)
      LeftBound160142.bound (LeftBound160142.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160143RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160142.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160142.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160145 .coefficient)
      LeftBound160123.bound (LeftBound160123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160128RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160123.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound160142.bound, LeftBound160123.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound160142.bound, LeftBound160123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound160142.actual selector witness, LeftBound160123.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound160146

namespace LeftBound160159
def owner : Owner := ⟨.program ⟨257⟩, ⟨41911⟩⟩
def transferEvent : Nat := 160159
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 160157 .coefficient, .predecessor 1 160158 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160157 .coefficient)
      LeftBound159988.bound (LeftBound159988.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160156RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159988.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159988.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160158 .coefficient)
      LeftBound159971.bound (LeftBound159971.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events624.exact159978RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound159971.bound, RecordedBoundRefines] <;> decide)
      (LeftBound159971.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159988.bound, LeftBound159971.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159988.bound, LeftBound159971.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159988.actual selector witness, LeftBound159971.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound160159

namespace LeftBound160162
def owner : Owner := ⟨.program ⟨257⟩, ⟨41911⟩⟩
def transferEvent : Nat := 160162
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 160156 .summary, .result 159978 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160156 .summary)
      LeftBound159990.bound (LeftBound159990.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨40795⟩⟩) (rawTerms := some (Proof.Events625.exact160156RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 159978 .summary)
      LeftBound159973.bound (LeftBound159973.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41910⟩⟩) (rawTerms := some (Proof.Events624.exact159978RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound159973.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound159990.bound, LeftBound159973.bound]
def bound : CoeffClass := .finite ⟨32193129122288829188810200055808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound159990.bound, LeftBound159973.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound159990.actual selector witness, LeftBound159973.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound160162

namespace LeftBound160166
def owner : Owner := ⟨.program ⟨257⟩, ⟨41912⟩⟩
def transferEvent : Nat := 160166
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 160164 .coefficient) (.predecessor 1 160165 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160164 .coefficient)
      LeftBound160159.bound (LeftBound160159.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160163RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160159.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160159.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160165 .coefficient)
      LeftBound15601.bound (LeftBound15601.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events060.exact15602RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15601.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15601.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound160159.bound LeftBound15601.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound160159.bound, LeftBound15601.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound160159.actual selector witness) * (LeftBound15601.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160166

namespace LeftBound160167
def owner : Owner := ⟨.program ⟨257⟩, ⟨41912⟩⟩
def transferEvent : Nat := 160167
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7159⟩⟩]⟩ [⟨.result 15598 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15598 .coefficient)
      LeftAuthority15597.bound (LeftAuthority15597.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7159⟩⟩) (rawTerms := some (Proof.Events060.exact15598RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15597.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15597.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15597.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15597.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15597.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound160167

namespace LeftBound160168
def owner : Owner := ⟨.program ⟨257⟩, ⟨41912⟩⟩
def transferEvent : Nat := 160168
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 160163 .summary) (.transfer 160167) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160163 .summary)
      LeftBound160162.bound (LeftBound160162.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨41911⟩⟩) (rawTerms := some (Proof.Events625.exact160163RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound160162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 160167)
      LeftBound160167.bound (LeftBound160167.actual selector witness) := by
  exact .transfer (LeftBound160167.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound160162.bound LeftBound160167.bound
def bound : CoeffClass := .finite ⟨345671091840339265080175045977281837137920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound160162.bound, LeftBound160167.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound160162.actual selector witness) * (LeftBound160167.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160168

namespace LeftBound160183
def owner : Owner := ⟨.program ⟨257⟩, ⟨39230⟩⟩
def transferEvent : Nat := 160183
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 160181 .coefficient) (.predecessor 1 160182 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160181 .coefficient)
      LeftBound151230.bound (LeftBound151230.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events590.exact151234RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound151230.bound, RecordedBoundRefines] <;> decide)
      (LeftBound151230.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160182 .coefficient)
      LeftAuthority160179.bound (LeftAuthority160179.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160180RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160179.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound151230.bound LeftAuthority160179.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151230.bound, LeftAuthority160179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound151230.actual selector witness) * (LeftAuthority160179.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160183

namespace LeftBound160184
def owner : Owner := ⟨.program ⟨257⟩, ⟨39230⟩⟩
def transferEvent : Nat := 160184
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨39228⟩⟩]⟩ [⟨.result 160180 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 160180 .coefficient)
      LeftAuthority160179.bound (LeftAuthority160179.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨39228⟩⟩) (rawTerms := some (Proof.Events625.exact160180RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160179.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160179.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority160179.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160179.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority160179.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound160184

namespace LeftBound160185
def owner : Owner := ⟨.program ⟨257⟩, ⟨39230⟩⟩
def transferEvent : Nat := 160185
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 151234 .summary) (.transfer 160184) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 151234 .summary)
      LeftBound151233.bound (LeftBound151233.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨38908⟩⟩) (rawTerms := some (Proof.Events590.exact151234RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound151233.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 160184)
      LeftBound160184.bound (LeftBound160184.actual selector witness) := by
  exact .transfer (LeftBound160184.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound151233.bound LeftBound160184.bound
def bound : CoeffClass := .finite ⟨32192736221397252361486566686720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound151233.bound, LeftBound160184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound151233.actual selector witness) * (LeftBound160184.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160185

namespace LeftBound160196
def owner : Owner := ⟨.program ⟨257⟩, ⟨38114⟩⟩
def transferEvent : Nat := 160196
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 160194 .coefficient) (.value (.predecessor 1 160195 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160194 .coefficient)
      LeftAuthority160192.bound (LeftAuthority160192.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority160192.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority160192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160195 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority160192.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority160192.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority160192.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound160196

namespace LeftBound160200
def owner : Owner := ⟨.program ⟨257⟩, ⟨38115⟩⟩
def transferEvent : Nat := 160200
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 160198 .coefficient) (.predecessor 1 160199 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 160198 .coefficient)
      LeftBound149117.bound (LeftBound149117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events582.exact149120RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound149117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound149117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 160199 .coefficient)
      LeftBound160196.bound (LeftBound160196.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events625.exact160197RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound160196.bound, RecordedBoundRefines] <;> decide)
      (LeftBound160196.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound149117.bound LeftBound160196.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound149117.bound, LeftBound160196.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound149117.actual selector witness) * (LeftBound160196.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound160200

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
