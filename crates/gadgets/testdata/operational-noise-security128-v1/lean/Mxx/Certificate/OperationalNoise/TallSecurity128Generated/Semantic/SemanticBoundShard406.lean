import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard098
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard099
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard376

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound65148
def owner : Owner := ⟨.program ⟨257⟩, ⟨25815⟩⟩
def transferEvent : Nat := 65148
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 65146 .coefficient) (.predecessor 1 65147 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65146 .coefficient)
      LeftAuthority2521.bound (LeftAuthority2521.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2522RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2521.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2521.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65147 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2521.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2521.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2521.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65148

namespace LeftBound65153
def owner : Owner := ⟨.program ⟨257⟩, ⟨10758⟩⟩
def transferEvent : Nat := 65153
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65151 .coefficient) (.predecessor 1 65152 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65151 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65152 .coefficient)
      LeftBound21087.bound (LeftBound21087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21087.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound21087.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound21087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound21087.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65153

namespace LeftBound65158
def owner : Owner := ⟨.program ⟨257⟩, ⟨25816⟩⟩
def transferEvent : Nat := 65158
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65156 .coefficient, .predecessor 1 65157 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65156 .coefficient)
      LeftBound65153.bound (LeftBound65153.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65155RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65153.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65153.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65157 .coefficient)
      LeftBound65148.bound (LeftBound65148.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65148.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65148.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65153.bound, LeftBound65148.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65153.bound, LeftBound65148.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65153.actual selector witness, LeftBound65148.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65158

namespace LeftBound65162
def owner : Owner := ⟨.program ⟨257⟩, ⟨25817⟩⟩
def transferEvent : Nat := 65162
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65160 .coefficient, .predecessor 1 65161 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65160 .coefficient)
      LeftBound65158.bound (LeftBound65158.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65159RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65158.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65158.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65161 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65158.bound, LeftBound21079.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65158.bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65158.actual selector witness, LeftBound21079.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65162

namespace LeftBound65163
def owner : Owner := ⟨.program ⟨257⟩, ⟨25817⟩⟩
def transferEvent : Nat := 65163
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨102⟩⟩]⟩ [⟨.result 21080 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21080 .coefficient)
      LeftBound21079.bound (LeftBound21079.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨102⟩⟩) (rawTerms := some (Proof.Events082.exact21080RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21079.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21079.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21079.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21079.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65163

namespace LeftBound65168
def owner : Owner := ⟨.program ⟨257⟩, ⟨65637⟩⟩
def transferEvent : Nat := 65168
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65166 .coefficient) (.predecessor 1 65167 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65166 .coefficient)
      LeftBound65162.bound (LeftBound65162.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65162.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65162.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65167 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound65162.bound LeftAuthority2524.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65162.bound, LeftAuthority2524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound65162.actual selector witness) * (LeftAuthority2524.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65168

namespace LeftBound65169
def owner : Owner := ⟨.program ⟨257⟩, ⟨65637⟩⟩
def transferEvent : Nat := 65169
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨65634⟩⟩], []⟩ [⟨.result 2525 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 2525 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨65634⟩⟩) (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority2524.bound []
def bound : CoeffClass := .finite ⟨28, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2524.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority2524.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65169

namespace LeftBound65170
def owner : Owner := ⟨.program ⟨257⟩, ⟨65637⟩⟩
def transferEvent : Nat := 65170
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65165 .summary) (.transfer 65169) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65165 .summary)
      LeftBound65163.bound (LeftBound65163.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨25817⟩⟩) (rawTerms := some (Proof.Events254.exact65165RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 65169)
      LeftBound65169.bound (LeftBound65169.actual selector witness) := by
  exact .transfer (LeftBound65169.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound65163.bound LeftBound65169.bound
def bound : CoeffClass := .finite ⟨23855104, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65163.bound, LeftBound65169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound65163.actual selector witness) * (LeftBound65169.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65170

namespace LeftBound65176
def owner : Owner := ⟨.program ⟨257⟩, ⟨65638⟩⟩
def transferEvent : Nat := 65176
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 65174 .coefficient) (.predecessor 1 65175 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65174 .coefficient)
      LeftAuthority2524.bound (LeftAuthority2524.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2525RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2524.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2524.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65175 .coefficient)
      LeftBound61276.bound (LeftBound61276.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events239.exact61278RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61276.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61276.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority2524.bound LeftBound61276.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority2524.bound, LeftBound61276.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority2524.actual selector witness) * (LeftBound61276.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound65176

namespace LeftBound65181
def owner : Owner := ⟨.program ⟨257⟩, ⟨10776⟩⟩
def transferEvent : Nat := 65181
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65179 .coefficient) (.predecessor 1 65180 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65179 .coefficient)
      LeftBound61147.bound (LeftBound61147.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events238.exact61148RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound61147.bound, RecordedBoundRefines] <;> decide)
      (LeftBound61147.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65180 .coefficient)
      LeftBound21128.bound (LeftBound21128.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21129RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21128.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21128.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound61147.bound LeftBound21128.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61147.bound, LeftBound21128.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound61147.actual selector witness) * (LeftBound21128.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65181

namespace LeftBound65186
def owner : Owner := ⟨.program ⟨257⟩, ⟨65639⟩⟩
def transferEvent : Nat := 65186
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65184 .coefficient, .predecessor 1 65185 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65184 .coefficient)
      LeftBound65181.bound (LeftBound65181.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65183RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65181.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65185 .coefficient)
      LeftBound65176.bound (LeftBound65176.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65178RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65176.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65181.bound, LeftBound65176.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65181.bound, LeftBound65176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65181.actual selector witness, LeftBound65176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65186

namespace LeftBound65190
def owner : Owner := ⟨.program ⟨257⟩, ⟨65640⟩⟩
def transferEvent : Nat := 65190
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 65188 .coefficient, .predecessor 1 65189 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65188 .coefficient)
      LeftBound65186.bound (LeftBound65186.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65187RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65186.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65186.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65189 .coefficient)
      LeftBound21120.bound (LeftBound21120.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21121RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21120.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound65186.bound, LeftBound21120.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65186.bound, LeftBound21120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound65186.actual selector witness, LeftBound21120.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound65190

namespace LeftBound65191
def owner : Owner := ⟨.program ⟨257⟩, ⟨65640⟩⟩
def transferEvent : Nat := 65191
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨120⟩⟩]⟩ [⟨.result 21121 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21121 .coefficient)
      LeftBound21120.bound (LeftBound21120.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨120⟩⟩) (rawTerms := some (Proof.Events082.exact21121RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21120.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21120.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound21120.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound21120.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound21120.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65191

namespace LeftBound65196
def owner : Owner := ⟨.program ⟨257⟩, ⟨65641⟩⟩
def transferEvent : Nat := 65196
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 65194 .coefficient) (.predecessor 1 65195 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 65194 .coefficient)
      LeftBound65190.bound (LeftBound65190.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events254.exact65193RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound65190.bound, RecordedBoundRefines] <;> decide)
      (LeftBound65190.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 65195 .coefficient)
      LeftBound21117.bound (LeftBound21117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events082.exact21118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound21117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound21117.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound65190.bound LeftBound21117.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65190.bound, LeftBound21117.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound65190.actual selector witness) * (LeftBound21117.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65196

namespace LeftBound65197
def owner : Owner := ⟨.program ⟨257⟩, ⟨65641⟩⟩
def transferEvent : Nat := 65197
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9541⟩⟩]⟩ [⟨.result 21114 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 21114 .coefficient)
      LeftAuthority21113.bound (LeftAuthority21113.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9541⟩⟩) (rawTerms := some (Proof.Events082.exact21114RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority21113.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority21113.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority21113.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority21113.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority21113.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound65197

namespace LeftBound65198
def owner : Owner := ⟨.program ⟨257⟩, ⟨65641⟩⟩
def transferEvent : Nat := 65198
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 65193 .summary) (.transfer 65197) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 65193 .summary)
      LeftBound65191.bound (LeftBound65191.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨65640⟩⟩) (rawTerms := some (Proof.Events254.exact65193RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound65191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 65197)
      LeftBound65197.bound (LeftBound65197.actual selector witness) := by
  exact .transfer (LeftBound65197.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound65191.bound LeftBound65197.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound65191.bound, LeftBound65197.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound65191.actual selector witness) * (LeftBound65197.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound65198

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
