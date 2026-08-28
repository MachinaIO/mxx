import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1693

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound251307
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251307
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26557⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 12611 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 613 .coefficient)
      LeftAuthority612.bound (LeftAuthority612.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6860⟩⟩) (rawTerms := some (Proof.Events002.exact613RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority612.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority612.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12611 .coefficient)
      LeftAuthority12610.bound (LeftAuthority12610.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26557⟩⟩) (rawTerms := some (Proof.Events049.exact12611RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12610.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12610.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority12610.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority12610.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority12610.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251307

namespace LeftBound251308
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251308
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251306, .transfer 251307]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251306)
      LeftBound251306.bound (LeftBound251306.actual selector witness) := by
  exact .transfer (LeftBound251306.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251307)
      LeftBound251307.bound (LeftBound251307.actual selector witness) := by
  exact .transfer (LeftBound251307.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251306.bound, LeftBound251307.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251306.bound, LeftBound251307.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251306.actual selector witness, LeftBound251307.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251308

namespace LeftBound251309
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251309
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨66238⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 12619 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 623 .coefficient)
      LeftAuthority622.bound (LeftAuthority622.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6870⟩⟩) (rawTerms := some (Proof.Events002.exact623RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority622.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority622.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12619 .coefficient)
      LeftAuthority12618.bound (LeftAuthority12618.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨66238⟩⟩) (rawTerms := some (Proof.Events049.exact12619RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12618.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority12618.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority12618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority12618.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251309

namespace LeftBound251310
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251310
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251308, .transfer 251309]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251308)
      LeftBound251308.bound (LeftBound251308.actual selector witness) := by
  exact .transfer (LeftBound251308.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251309)
      LeftBound251309.bound (LeftBound251309.actual selector witness) := by
  exact .transfer (LeftBound251309.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251308.bound, LeftBound251309.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251308.bound, LeftBound251309.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251308.actual selector witness, LeftBound251309.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251310

namespace LeftBound251311
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251311
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨62990⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 12627 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 633 .coefficient)
      LeftAuthority632.bound (LeftAuthority632.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6732⟩⟩) (rawTerms := some (Proof.Events002.exact633RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority632.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority632.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12627 .coefficient)
      LeftAuthority12626.bound (LeftAuthority12626.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨62990⟩⟩) (rawTerms := some (Proof.Events049.exact12627RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12626.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12626.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority12626.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority12626.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority12626.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251311

namespace LeftBound251312
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251312
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251310, .transfer 251311]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251310)
      LeftBound251310.bound (LeftBound251310.actual selector witness) := by
  exact .transfer (LeftBound251310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251311)
      LeftBound251311.bound (LeftBound251311.actual selector witness) := by
  exact .transfer (LeftBound251311.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251310.bound, LeftBound251311.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251310.bound, LeftBound251311.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251310.actual selector witness, LeftBound251311.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251312

namespace LeftBound251313
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251313
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60010⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 12635 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 643 .coefficient)
      LeftAuthority642.bound (LeftAuthority642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6736⟩⟩) (rawTerms := some (Proof.Events002.exact643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority642.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12635 .coefficient)
      LeftAuthority12634.bound (LeftAuthority12634.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨60010⟩⟩) (rawTerms := some (Proof.Events049.exact12635RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12634.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12634.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority12634.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority12634.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority12634.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251313

namespace LeftBound251314
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251314
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251312, .transfer 251313]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251312)
      LeftBound251312.bound (LeftBound251312.actual selector witness) := by
  exact .transfer (LeftBound251312.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251313)
      LeftBound251313.bound (LeftBound251313.actual selector witness) := by
  exact .transfer (LeftBound251313.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251312.bound, LeftBound251313.bound]
def bound : CoeffClass := .finite ⟨6729770197368168063193080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251312.bound, LeftBound251313.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251312.actual selector witness, LeftBound251313.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251314

namespace LeftBound251315
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251315
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57030⟩⟩], []⟩ [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 12643 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 653 .coefficient)
      LeftAuthority652.bound (LeftAuthority652.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6741⟩⟩) (rawTerms := some (Proof.Events002.exact653RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority652.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority652.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12643 .coefficient)
      LeftAuthority12642.bound (LeftAuthority12642.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨57030⟩⟩) (rawTerms := some (Proof.Events049.exact12643RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12642.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12642.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority652.bound [LeftAuthority12642.bound]
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority652.bound, LeftAuthority12642.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority652.actual selector witness) * ([LeftAuthority12642.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251315

namespace LeftBound251316
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251316
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251314, .transfer 251315]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251314)
      LeftBound251314.bound (LeftBound251314.actual selector witness) := by
  exact .transfer (LeftBound251314.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251315)
      LeftBound251315.bound (LeftBound251315.actual selector witness) := by
  exact .transfer (LeftBound251315.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251314.bound, LeftBound251315.bound]
def bound : CoeffClass := .finite ⟨6950548326985875302691000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251314.bound, LeftBound251315.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251314.actual selector witness, LeftBound251315.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251316

namespace LeftBound251317
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251317
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54050⟩⟩], []⟩ [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 12651 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 663 .coefficient)
      LeftAuthority662.bound (LeftAuthority662.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6757⟩⟩) (rawTerms := some (Proof.Events002.exact663RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority662.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority662.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12651 .coefficient)
      LeftAuthority12650.bound (LeftAuthority12650.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54050⟩⟩) (rawTerms := some (Proof.Events049.exact12651RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12650.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12650.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority662.bound [LeftAuthority12650.bound]
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority662.bound, LeftAuthority12650.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority662.actual selector witness) * ([LeftAuthority12650.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251317

namespace LeftBound251318
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251318
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251316, .transfer 251317]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251316)
      LeftBound251316.bound (LeftBound251316.actual selector witness) := by
  exact .transfer (LeftBound251316.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251317)
      LeftBound251317.bound (LeftBound251317.actual selector witness) := by
  exact .transfer (LeftBound251317.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251316.bound, LeftBound251317.bound]
def bound : CoeffClass := .finite ⟨7167080723341703556813960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251316.bound, LeftBound251317.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251316.actual selector witness, LeftBound251317.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251318

namespace LeftBound251319
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251319
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6768⟩⟩, ⟨.program ⟨257⟩, ⟨51070⟩⟩], []⟩ [⟨.result 673 .coefficient, true, some 1⟩, ⟨.result 12659 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 673 .coefficient)
      LeftAuthority672.bound (LeftAuthority672.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6768⟩⟩) (rawTerms := some (Proof.Events002.exact673RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority672.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority672.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12659 .coefficient)
      LeftAuthority12658.bound (LeftAuthority12658.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨51070⟩⟩) (rawTerms := some (Proof.Events049.exact12659RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12658.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12658.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority672.bound [LeftAuthority12658.bound]
def bound : CoeffClass := .finite ⟨213251602471649038151400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority672.bound, LeftAuthority12658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority672.actual selector witness) * ([LeftAuthority12658.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251319

namespace LeftBound251320
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251320
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251318, .transfer 251319]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251318)
      LeftBound251318.bound (LeftBound251318.actual selector witness) := by
  exact .transfer (LeftBound251318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251319)
      LeftBound251319.bound (LeftBound251319.actual selector witness) := by
  exact .transfer (LeftBound251319.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251318.bound, LeftBound251319.bound]
def bound : CoeffClass := .finite ⟨7380332325813352594965360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251318.bound, LeftBound251319.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251318.actual selector witness, LeftBound251319.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251320

namespace LeftBound251321
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251321
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6794⟩⟩, ⟨.program ⟨257⟩, ⟨32006⟩⟩], []⟩ [⟨.result 683 .coefficient, true, some 1⟩, ⟨.result 12667 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 683 .coefficient)
      LeftAuthority682.bound (LeftAuthority682.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6794⟩⟩) (rawTerms := some (Proof.Events002.exact683RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority682.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority682.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 12667 .coefficient)
      LeftAuthority12666.bound (LeftAuthority12666.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨32006⟩⟩) (rawTerms := some (Proof.Events049.exact12667RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12666.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12666.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority682.bound [LeftAuthority12666.bound]
def bound : CoeffClass := .finite ⟨201065796616126235971320, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority682.bound, LeftAuthority12666.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority682.actual selector witness) * ([LeftAuthority12666.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound251321

namespace LeftBound251322
def owner : Owner := ⟨.program ⟨257⟩, ⟨67369⟩⟩
def transferEvent : Nat := 251322
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 251320, .transfer 251321]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251320)
      LeftBound251320.bound (LeftBound251320.actual selector witness) := by
  exact .transfer (LeftBound251320.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 251321)
      LeftBound251321.bound (LeftBound251321.actual selector witness) := by
  exact .transfer (LeftBound251321.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound251320.bound, LeftBound251321.bound]
def bound : CoeffClass := .finite ⟨7581398122429478830936680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound251320.bound, LeftBound251321.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound251320.actual selector witness, LeftBound251321.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound251322

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
