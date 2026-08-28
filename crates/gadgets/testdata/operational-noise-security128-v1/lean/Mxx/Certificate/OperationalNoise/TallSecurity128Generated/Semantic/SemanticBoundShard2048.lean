import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard000
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard122
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2047

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound301207
def owner : Owner := ⟨.program ⟨257⟩, ⟨24173⟩⟩
def transferEvent : Nat := 301207
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301205 .coefficient, .predecessor 1 301206 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301205 .coefficient)
      LeftBound301203.bound (LeftBound301203.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301204RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301203.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301203.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301206 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301203.bound, LeftBound24085.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301203.bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301203.actual selector witness, LeftBound24085.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301207

namespace LeftBound301208
def owner : Owner := ⟨.program ⟨257⟩, ⟨24173⟩⟩
def transferEvent : Nat := 301208
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨133⟩⟩]⟩ [⟨.result 24086 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24086 .coefficient)
      LeftBound24085.bound (LeftBound24085.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨133⟩⟩) (rawTerms := some (Proof.Events094.exact24086RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24085.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24085.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24085.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24085.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24085.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301208

namespace LeftBound301213
def owner : Owner := ⟨.program ⟨257⟩, ⟨31218⟩⟩
def transferEvent : Nat := 301213
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301211 .coefficient) (.predecessor 1 301212 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301211 .coefficient)
      LeftBound301207.bound (LeftBound301207.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301210RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301207.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301207.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301212 .coefficient)
      LeftAuthority14614.bound (LeftAuthority14614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14614.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftBound301207.bound LeftAuthority14614.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301207.bound, LeftAuthority14614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftBound301207.actual selector witness) * (LeftAuthority14614.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301213

namespace LeftBound301214
def owner : Owner := ⟨.program ⟨257⟩, ⟨31218⟩⟩
def transferEvent : Nat := 301214
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨31215⟩⟩], []⟩ [⟨.result 14615 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 14615 .coefficient)
      LeftAuthority14614.bound (LeftAuthority14614.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨31215⟩⟩) (rawTerms := some (Proof.Events057.exact14615RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14614.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority14614.bound []
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority14614.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301214

namespace LeftBound301215
def owner : Owner := ⟨.program ⟨257⟩, ⟨31218⟩⟩
def transferEvent : Nat := 301215
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 301210 .summary) (.transfer 301214) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301210 .summary)
      LeftBound301208.bound (LeftBound301208.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨24173⟩⟩) (rawTerms := some (Proof.Events1176.exact301210RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301208.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 301214)
      LeftBound301214.bound (LeftBound301214.actual selector witness) := by
  exact .transfer (LeftBound301214.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound301208.bound LeftBound301214.bound
def bound : CoeffClass := .finite ⟨5111808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301208.bound, LeftBound301214.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound301208.actual selector witness) * (LeftBound301214.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301215

namespace LeftBound301221
def owner : Owner := ⟨.program ⟨257⟩, ⟨31219⟩⟩
def transferEvent : Nat := 301221
def frameStart : Nat := 0
def rule : BoundRule := .tensor (.predecessor 0 301219 .coefficient) (.predecessor 1 301220 .coefficient) true false
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301219 .coefficient)
      LeftAuthority14614.bound (LeftAuthority14614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events057.exact14615RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority14614.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority14614.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301220 .coefficient)
      LeftBound30.bound (LeftBound30.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact32RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound30.bound, RecordedBoundRefines] <;> decide)
      (LeftBound30.derived selector witness)

def rawBound : CoeffClass := tensorWithFacts 32768 ⟨true, false, none, none, none⟩ LeftAuthority14614.bound LeftBound30.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority14614.bound, LeftBound30.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := tensorFactor 32768 ⟨true, false, none, none, none⟩ * (LeftAuthority14614.actual selector witness) * (LeftBound30.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .tensor (by rfl) (input0 selector witness) (input1 selector witness)
end LeftBound301221

namespace LeftBound301226
def owner : Owner := ⟨.program ⟨257⟩, ⟨7435⟩⟩
def transferEvent : Nat := 301226
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301224 .coefficient) (.predecessor 1 301225 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301224 .coefficient)
      LeftBound26.bound (LeftBound26.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact27RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound26.bound, RecordedBoundRefines] <;> decide)
      (LeftBound26.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301225 .coefficient)
      LeftBound24134.bound (LeftBound24134.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24135RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24134.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24134.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound26.bound LeftBound24134.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound26.bound, LeftBound24134.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound26.actual selector witness) * (LeftBound24134.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 40) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301226

namespace LeftBound301231
def owner : Owner := ⟨.program ⟨257⟩, ⟨31220⟩⟩
def transferEvent : Nat := 301231
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301229 .coefficient, .predecessor 1 301230 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301229 .coefficient)
      LeftBound301226.bound (LeftBound301226.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301228RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301226.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301226.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301230 .coefficient)
      LeftBound301221.bound (LeftBound301221.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301223RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301221.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301221.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301226.bound, LeftBound301221.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301226.bound, LeftBound301221.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301226.actual selector witness, LeftBound301221.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301231

namespace LeftBound301235
def owner : Owner := ⟨.program ⟨257⟩, ⟨31221⟩⟩
def transferEvent : Nat := 301235
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301233 .coefficient, .predecessor 1 301234 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301233 .coefficient)
      LeftBound301231.bound (LeftBound301231.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301232RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301231.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301231.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301234 .coefficient)
      LeftBound24126.bound (LeftBound24126.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24127RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24126.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301231.bound, LeftBound24126.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301231.bound, LeftBound24126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301231.actual selector witness, LeftBound24126.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301235

namespace LeftBound301236
def owner : Owner := ⟨.program ⟨257⟩, ⟨31221⟩⟩
def transferEvent : Nat := 301236
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨113⟩⟩]⟩ [⟨.result 24127 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24127 .coefficient)
      LeftBound24126.bound (LeftBound24126.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨113⟩⟩) (rawTerms := some (Proof.Events094.exact24127RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24126.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24126.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftBound24126.bound []
def bound : CoeffClass := .finite ⟨26, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound24126.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftBound24126.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301236

namespace LeftBound301241
def owner : Owner := ⟨.program ⟨257⟩, ⟨31222⟩⟩
def transferEvent : Nat := 301241
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301239 .coefficient) (.predecessor 1 301240 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301239 .coefficient)
      LeftBound301235.bound (LeftBound301235.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301238RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301235.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301235.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301240 .coefficient)
      LeftBound24123.bound (LeftBound24123.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events094.exact24124RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound24123.bound, RecordedBoundRefines] <;> decide)
      (LeftBound24123.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301235.bound LeftBound24123.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301235.bound, LeftBound24123.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301235.actual selector witness) * (LeftBound24123.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301241

namespace LeftBound301242
def owner : Owner := ⟨.program ⟨257⟩, ⟨31222⟩⟩
def transferEvent : Nat := 301242
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨9577⟩⟩]⟩ [⟨.result 24120 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 24120 .coefficient)
      LeftAuthority24119.bound (LeftAuthority24119.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨9577⟩⟩) (rawTerms := some (Proof.Events094.exact24120RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority24119.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority24119.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority24119.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority24119.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority24119.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound301242

namespace LeftBound301243
def owner : Owner := ⟨.program ⟨257⟩, ⟨31222⟩⟩
def transferEvent : Nat := 301243
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 301238 .summary) (.transfer 301242) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301238 .summary)
      LeftBound301236.bound (LeftBound301236.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31221⟩⟩) (rawTerms := some (Proof.Events1176.exact301238RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301236.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 301242)
      LeftBound301242.bound (LeftBound301242.actual selector witness) := by
  exact .transfer (LeftBound301242.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301236.bound LeftBound301242.bound
def bound : CoeffClass := .finite ⟨279172874240, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301236.bound, LeftBound301242.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301236.actual selector witness) * (LeftBound301242.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301243

namespace LeftBound301251
def owner : Owner := ⟨.program ⟨257⟩, ⟨31223⟩⟩
def transferEvent : Nat := 301251
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 301249 .coefficient, .predecessor 1 301250 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301249 .coefficient)
      LeftBound301241.bound (LeftBound301241.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301248RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301241.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301241.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301250 .coefficient)
      LeftBound301213.bound (LeftBound301213.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301218RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301213.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301213.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301241.bound, LeftBound301213.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301241.bound, LeftBound301213.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301241.actual selector witness, LeftBound301213.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301251

namespace LeftBound301253
def owner : Owner := ⟨.program ⟨257⟩, ⟨31223⟩⟩
def transferEvent : Nat := 301253
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 301248 .summary, .result 301218 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301248 .summary)
      LeftBound301243.bound (LeftBound301243.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31222⟩⟩) (rawTerms := some (Proof.Events1176.exact301248RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301243.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 301218 .summary)
      LeftBound301215.bound (LeftBound301215.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨31218⟩⟩) (rawTerms := some (Proof.Events1176.exact301218RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound301215.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound301243.bound, LeftBound301215.bound]
def bound : CoeffClass := .finite ⟨279177986048, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301243.bound, LeftBound301215.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound301243.actual selector witness, LeftBound301215.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound301253

namespace LeftBound301257
def owner : Owner := ⟨.program ⟨257⟩, ⟨33350⟩⟩
def transferEvent : Nat := 301257
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 301255 .coefficient) (.predecessor 1 301256 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 301255 .coefficient)
      LeftBound301251.bound (LeftBound301251.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301254RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound301251.bound, RecordedBoundRefines] <;> decide)
      (LeftBound301251.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 301256 .coefficient)
      LeftAuthority301189.bound (LeftAuthority301189.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1176.exact301190RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority301189.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority301189.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound301251.bound LeftAuthority301189.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound301251.bound, LeftAuthority301189.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound301251.actual selector witness) * (LeftAuthority301189.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound301257

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
