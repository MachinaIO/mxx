import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard276
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard331

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound54117
def owner : Owner := ⟨.program ⟨257⟩, ⟨23241⟩⟩
def transferEvent : Nat := 54117
def frameStart : Nat := 54030
def rule : BoundRule := .sum [.predecessor 0 54115 .coefficient, .predecessor 1 54116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54115 .coefficient)
      LeftBound54112.bound (LeftBound54112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54112.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54116 .coefficient)
      LeftBound54089.bound (LeftBound54089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54091RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54089.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54112.bound, LeftBound54089.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54112.bound, LeftBound54089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54112.actual selector witness, LeftBound54089.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54117

namespace LeftBound54121
def owner : Owner := ⟨.program ⟨257⟩, ⟨23530⟩⟩
def transferEvent : Nat := 54121
def frameStart : Nat := 54030
def rule : BoundRule := .product (.predecessor 0 54119 .coefficient) (.predecessor 1 54120 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54119 .coefficient)
      LeftBound54117.bound (LeftBound54117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54120 .coefficient)
      LeftAuthority54074.bound (LeftAuthority54074.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54075RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54074.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54074.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound54117.bound LeftAuthority54074.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54117.bound, LeftAuthority54074.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound54117.actual selector witness) * (LeftAuthority54074.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54121

namespace LeftBound54132
def owner : Owner := ⟨.program ⟨257⟩, ⟨21874⟩⟩
def transferEvent : Nat := 54132
def frameStart : Nat := 54030
def rule : BoundRule := .product (.predecessor 0 54130 .coefficient) (.predecessor 1 54131 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54130 .coefficient)
      LeftAuthority54085.bound (LeftAuthority54085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54085.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54085.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54131 .coefficient)
      LeftAuthority54128.bound (LeftAuthority54128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54128.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority54085.bound LeftAuthority54128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54085.bound, LeftAuthority54128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority54085.actual selector witness) * (LeftAuthority54128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54132

namespace LeftBound54140
def owner : Owner := ⟨.program ⟨257⟩, ⟨21875⟩⟩
def transferEvent : Nat := 54140
def frameStart : Nat := 54030
def rule : BoundRule := .sum [.predecessor 0 54138 .coefficient, .predecessor 1 54139 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54138 .coefficient)
      LeftAuthority54136.bound (LeftAuthority54136.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54137RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54136.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54136.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54139 .coefficient)
      LeftBound54132.bound (LeftBound54132.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54132.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54132.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority54136.bound, LeftBound54132.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54136.bound, LeftBound54132.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority54136.actual selector witness, LeftBound54132.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54140

namespace LeftBound54144
def owner : Owner := ⟨.program ⟨257⟩, ⟨23531⟩⟩
def transferEvent : Nat := 54144
def frameStart : Nat := 54030
def rule : BoundRule := .sum [.predecessor 0 54142 .coefficient, .predecessor 1 54143 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54142 .coefficient)
      LeftBound54140.bound (LeftBound54140.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54141RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54140.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54140.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54143 .coefficient)
      LeftBound54121.bound (LeftBound54121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54121.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54140.bound, LeftBound54121.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54140.bound, LeftBound54121.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54140.actual selector witness, LeftBound54121.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54144

namespace LeftBound54157
def owner : Owner := ⟨.program ⟨257⟩, ⟨23529⟩⟩
def transferEvent : Nat := 54157
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 54155 .coefficient, .predecessor 1 54156 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54155 .coefficient)
      LeftBound53978.bound (LeftBound53978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54154RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54156 .coefficient)
      LeftBound53961.bound (LeftBound53961.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound53961.bound, RecordedBoundRefines] <;> decide)
      (LeftBound53961.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53978.bound, LeftBound53961.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53978.bound, LeftBound53961.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53978.actual selector witness, LeftBound53961.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54157

namespace LeftBound54160
def owner : Owner := ⟨.program ⟨257⟩, ⟨23529⟩⟩
def transferEvent : Nat := 54160
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 54154 .summary, .result 53968 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54154 .summary)
      LeftBound53980.bound (LeftBound53980.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨22452⟩⟩) (rawTerms := some (Proof.Events211.exact54154RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53980.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53968 .summary)
      LeftBound53963.bound (LeftBound53963.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23528⟩⟩) (rawTerms := some (Proof.Events210.exact53968RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound53963.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound53980.bound, LeftBound53963.bound]
def bound : CoeffClass := .finite ⟨2997834576566628384768, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound53980.bound, LeftBound53963.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound53980.actual selector witness, LeftBound53963.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54160

namespace LeftBound54164
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def transferEvent : Nat := 54164
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54162 .coefficient) (.predecessor 1 54163 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54162 .coefficient)
      LeftBound54157.bound (LeftBound54157.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54161RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54157.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54157.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54163 .coefficient)
      LeftAuthority53883.bound (LeftAuthority53883.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events210.exact53884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53883.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound54157.bound LeftAuthority53883.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54157.bound, LeftAuthority53883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound54157.actual selector witness) * (LeftAuthority53883.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54164

namespace LeftBound54165
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def transferEvent : Nat := 54165
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨24120⟩⟩]⟩ [⟨.result 53884 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 53884 .coefficient)
      LeftAuthority53883.bound (LeftAuthority53883.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨24120⟩⟩) (rawTerms := some (Proof.Events210.exact53884RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority53883.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority53883.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority53883.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority53883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority53883.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54165

namespace LeftBound54166
def owner : Owner := ⟨.program ⟨257⟩, ⟨24122⟩⟩
def transferEvent : Nat := 54166
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 54161 .summary) (.transfer 54165) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54161 .summary)
      LeftBound54160.bound (LeftBound54160.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨23529⟩⟩) (rawTerms := some (Proof.Events211.exact54161RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound54160.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 54165)
      LeftBound54165.bound (LeftBound54165.actual selector witness) := by
  exact .transfer (LeftBound54165.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound54160.bound LeftBound54165.bound
def bound : CoeffClass := .finite ⟨32189003662929192193909661368320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54160.bound, LeftBound54165.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound54160.actual selector witness) * (LeftBound54165.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54166

namespace LeftBound54177
def owner : Owner := ⟨.program ⟨257⟩, ⟨22838⟩⟩
def transferEvent : Nat := 54177
def frameStart : Nat := 0
def rule : BoundRule := .scale (.predecessor 0 54175 .coefficient) (.value (.predecessor 1 54176 .coefficient))
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54175 .coefficient)
      LeftAuthority54173.bound (LeftAuthority54173.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54173.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54176 .coefficient)
      LeftAuthority3.bound (LeftAuthority3.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority3.derived selector witness)

def rawBound : CoeffClass := scaleValue LeftAuthority54173.bound LeftAuthority3.bound
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54173.bound, LeftAuthority3.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority54173.actual selector witness) * (LeftAuthority3.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .scaleValue (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound54177

namespace LeftBound54181
def owner : Owner := ⟨.program ⟨257⟩, ⟨22839⟩⟩
def transferEvent : Nat := 54181
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 54179 .coefficient) (.predecessor 1 54180 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54179 .coefficient)
      LeftBound46742.bound (LeftBound46742.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound46742.bound, RecordedBoundRefines] <;> decide)
      (LeftBound46742.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54180 .coefficient)
      LeftBound54177.bound (LeftBound54177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events211.exact54178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound54177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound54177.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46742.bound LeftBound54177.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46742.bound, LeftBound54177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46742.actual selector witness) * (LeftBound54177.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54181

namespace LeftBound54182
def owner : Owner := ⟨.program ⟨257⟩, ⟨22839⟩⟩
def transferEvent : Nat := 54182
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨22836⟩⟩]⟩ [⟨.result 54174 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 54174 .coefficient)
      LeftAuthority54173.bound (LeftAuthority54173.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨22836⟩⟩) (rawTerms := some (Proof.Events211.exact54174RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54173.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54173.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority54173.bound []
def bound : CoeffClass := .finite ⟨5647228698, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54173.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority54173.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound54182

namespace LeftBound54183
def owner : Owner := ⟨.program ⟨257⟩, ⟨22839⟩⟩
def transferEvent : Nat := 54183
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 46745 .summary) (.transfer 54182) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 46745 .summary)
      LeftBound46743.bound (LeftBound46743.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨11216⟩⟩) (rawTerms := some (Proof.Events182.exact46745RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound46743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 54182)
      LeftBound54182.bound (LeftBound54182.actual selector witness) := by
  exact .transfer (LeftBound54182.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1376256 LeftBound46743.bound LeftBound54182.bound
def bound : CoeffClass := .finite ⟨202072841853861888, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound46743.bound, LeftBound54182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1376256 * (LeftBound46743.actual selector witness) * (LeftBound54182.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 42) (rightRows := 42) (rightColumns := 40) (ringDimension := 32768) (factor := 1376256) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound54183

namespace LeftBound54278
def owner : Owner := ⟨.program ⟨257⟩, ⟨21873⟩⟩
def transferEvent : Nat := 54278
def frameStart : Nat := 54239
def rule : BoundRule := .identity (.predecessor 0 54277 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54277 .coefficient)
      LeftAuthority54275.bound (LeftAuthority54275.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events212.exact54276RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority54275.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority54275.derived selector witness)

def rawBound : CoeffClass := LeftAuthority54275.bound
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority54275.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority54275.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound54278

namespace LeftBound54295
def owner : Owner := ⟨.program ⟨257⟩, ⟨23318⟩⟩
def transferEvent : Nat := 54295
def frameStart : Nat := 54239
def rule : BoundRule := .sum [.predecessor 0 54293 .coefficient, .predecessor 1 54294 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 54293 .coefficient)
      LeftBound54278.bound (LeftBound54278.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound54278.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 54294 .coefficient)
      LeftAuthority54291.bound (LeftAuthority54291.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority54291.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound54278.bound, LeftAuthority54291.bound]
def bound : CoeffClass := .finite ⟨4, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound54278.bound, LeftAuthority54291.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound54278.actual selector witness, LeftAuthority54291.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound54295

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
