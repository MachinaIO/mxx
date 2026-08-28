import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard071
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1084
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1086
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1090

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound164169
def owner : Owner := ⟨.program ⟨257⟩, ⟨45256⟩⟩
def transferEvent : Nat := 164169
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164167 .coefficient) (.predecessor 1 164168 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164167 .coefficient)
      LeftBound164163.bound (LeftBound164163.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164166RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164163.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164163.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164168 .coefficient)
      LeftAuthority7599.bound (LeftAuthority7599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7599.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound164163.bound LeftAuthority7599.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164163.bound, LeftAuthority7599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound164163.actual selector witness) * (LeftAuthority7599.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164169

namespace LeftBound164170
def owner : Owner := ⟨.program ⟨257⟩, ⟨45256⟩⟩
def transferEvent : Nat := 164170
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨14841⟩⟩], []⟩ [⟨.result 7600 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 7600 .coefficient)
      LeftAuthority7599.bound (LeftAuthority7599.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨14841⟩⟩) (rawTerms := some (Proof.Events029.exact7600RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7599.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority7599.bound []
def bound : CoeffClass := .finite ⟨58, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7599.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority7599.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164170

namespace LeftBound164171
def owner : Owner := ⟨.program ⟨257⟩, ⟨45256⟩⟩
def transferEvent : Nat := 164171
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 164166 .summary) (.transfer 164170) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164166 .summary)
      LeftBound164164.bound (LeftBound164164.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45255⟩⟩) (rawTerms := some (Proof.Events641.exact164166RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164164.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 164170)
      LeftBound164170.bound (LeftBound164170.actual selector witness) := by
  exact .transfer (LeftBound164170.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound164164.bound LeftBound164170.bound
def bound : CoeffClass := .finite ⟨49414144, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164164.bound, LeftBound164170.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound164164.actual selector witness) * (LeftBound164170.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164171

namespace LeftBound164177
def owner : Owner := ⟨.program ⟨257⟩, ⟨14842⟩⟩
def transferEvent : Nat := 164177
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 164175 .coefficient) (.predecessor 1 164176 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164175 .coefficient)
      LeftAuthority7599.bound (LeftAuthority7599.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events029.exact7600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority7599.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority7599.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164176 .coefficient)
      LeftBound163651.bound (LeftBound163651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events639.exact163653RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163651.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163651.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority7599.bound LeftBound163651.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority7599.bound, LeftBound163651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority7599.actual selector witness) * (LeftBound163651.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound164177

namespace LeftBound164182
def owner : Owner := ⟨.program ⟨257⟩, ⟨9063⟩⟩
def transferEvent : Nat := 164182
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164180 .coefficient) (.predecessor 1 164181 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164180 .coefficient)
      LeftBound163522.bound (LeftBound163522.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events638.exact163523RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound163522.bound, RecordedBoundRefines] <;> decide)
      (LeftBound163522.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164181 .coefficient)
      LeftBound17621.bound (LeftBound17621.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17622RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17621.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17621.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound163522.bound LeftBound17621.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound163522.bound, LeftBound17621.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound163522.actual selector witness) * (LeftBound17621.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164182

namespace LeftBound164187
def owner : Owner := ⟨.program ⟨257⟩, ⟨14843⟩⟩
def transferEvent : Nat := 164187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 164185 .coefficient, .predecessor 1 164186 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164185 .coefficient)
      LeftBound164182.bound (LeftBound164182.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164184RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164182.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164182.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164186 .coefficient)
      LeftBound164177.bound (LeftBound164177.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164179RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164177.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164177.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164182.bound, LeftBound164177.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164182.bound, LeftBound164177.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164182.actual selector witness, LeftBound164177.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164187

namespace LeftBound164191
def owner : Owner := ⟨.program ⟨257⟩, ⟨14844⟩⟩
def transferEvent : Nat := 164191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 164189 .coefficient, .predecessor 1 164190 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164189 .coefficient)
      LeftBound164187.bound (LeftBound164187.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164188RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164187.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164190 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164187.bound, LeftBound17613.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164187.bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164187.actual selector witness, LeftBound17613.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164191

namespace LeftBound164192
def owner : Owner := ⟨.program ⟨257⟩, ⟨14844⟩⟩
def transferEvent : Nat := 164192
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨127⟩⟩]⟩ [⟨.result 17614 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17614 .coefficient)
      LeftBound17613.bound (LeftBound17613.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨127⟩⟩) (rawTerms := some (Proof.Events068.exact17614RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17613.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17613.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound17613.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound17613.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound17613.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164192

namespace LeftBound164197
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def transferEvent : Nat := 164197
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164195 .coefficient) (.predecessor 1 164196 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164195 .coefficient)
      LeftBound164191.bound (LeftBound164191.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164194RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164191.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164191.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164196 .coefficient)
      LeftBound17610.bound (LeftBound17610.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events068.exact17611RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound17610.bound, RecordedBoundRefines] <;> decide)
      (LeftBound17610.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164191.bound LeftBound17610.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164191.bound, LeftBound17610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164191.actual selector witness) * (LeftBound17610.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164197

namespace LeftBound164198
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def transferEvent : Nat := 164198
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩ [⟨.result 17607 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 17607 .coefficient)
      LeftAuthority17606.bound (LeftAuthority17606.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9562⟩⟩) (rawTerms := some (Proof.Events068.exact17607RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority17606.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority17606.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority17606.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority17606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority17606.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164198

namespace LeftBound164199
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def transferEvent : Nat := 164199
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 164194 .summary) (.transfer 164198) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164194 .summary)
      LeftBound164192.bound (LeftBound164192.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14844⟩⟩) (rawTerms := some (Proof.Events641.exact164194RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164192.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 164198)
      LeftBound164198.bound (LeftBound164198.actual selector witness) := by
  exact .transfer (LeftBound164198.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164192.bound LeftBound164198.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164192.bound, LeftBound164198.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164192.actual selector witness) * (LeftBound164198.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164199

namespace LeftBound164207
def owner : Owner := ⟨.program ⟨257⟩, ⟨45257⟩⟩
def transferEvent : Nat := 164207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 164205 .coefficient, .predecessor 1 164206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164205 .coefficient)
      LeftBound164197.bound (LeftBound164197.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164197.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164197.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164206 .coefficient)
      LeftBound164169.bound (LeftBound164169.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164174RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164169.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164169.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164197.bound, LeftBound164169.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164197.bound, LeftBound164169.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164197.actual selector witness, LeftBound164169.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164207

namespace LeftBound164209
def owner : Owner := ⟨.program ⟨257⟩, ⟨45257⟩⟩
def transferEvent : Nat := 164209
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 164204 .summary, .result 164174 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164204 .summary)
      LeftBound164199.bound (LeftBound164199.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨14845⟩⟩) (rawTerms := some (Proof.Events641.exact164204RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164199.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164174 .summary)
      LeftBound164171.bound (LeftBound164171.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45256⟩⟩) (rawTerms := some (Proof.Events641.exact164174RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164171.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound164199.bound, LeftBound164171.bound]
def bound : CoeffClass := .finite ⟨279222288384, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164199.bound, LeftBound164171.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound164199.actual selector witness, LeftBound164171.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound164209

namespace LeftBound164213
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def transferEvent : Nat := 164213
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 164211 .coefficient) (.predecessor 1 164212 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 164211 .coefficient)
      LeftBound164207.bound (LeftBound164207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound164207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound164207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 164212 .coefficient)
      LeftAuthority164145.bound (LeftAuthority164145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events641.exact164146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164145.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164207.bound LeftAuthority164145.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164207.bound, LeftAuthority164145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164207.actual selector witness) * (LeftAuthority164145.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164213

namespace LeftBound164214
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def transferEvent : Nat := 164214
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨47023⟩⟩]⟩ [⟨.result 164146 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164146 .coefficient)
      LeftAuthority164145.bound (LeftAuthority164145.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨47023⟩⟩) (rawTerms := some (Proof.Events641.exact164146RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority164145.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority164145.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority164145.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority164145.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority164145.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound164214

namespace LeftBound164215
def owner : Owner := ⟨.program ⟨257⟩, ⟨47024⟩⟩
def transferEvent : Nat := 164215
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 164210 .summary) (.transfer 164214) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 164210 .summary)
      LeftBound164209.bound (LeftBound164209.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨45257⟩⟩) (rawTerms := some (Proof.Events641.exact164210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound164209.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 164214)
      LeftBound164214.bound (LeftBound164214.actual selector witness) := by
  exact .transfer (LeftBound164214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound164209.bound LeftBound164214.bound
def bound : CoeffClass := .finite ⟨2998126492308901724160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound164209.bound, LeftBound164214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound164209.actual selector witness) * (LeftBound164214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound164215

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
