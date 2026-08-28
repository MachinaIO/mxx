import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard374

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound61177
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61177
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61175, .transfer 61176]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61175)
      LeftBound61175.bound (LeftBound61175.actual selector witness) := by
  exact .transfer (LeftBound61175.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61176)
      LeftBound61176.bound (LeftBound61176.actual selector witness) := by
  exact .transfer (LeftBound61176.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61175.bound, LeftBound61176.bound]
def bound : CoeffClass := .finite ⟨5372571899340058642669440, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61175.bound, LeftBound61176.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61175.actual selector witness, LeftBound61176.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61177

namespace LeftBound61178
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61178
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨35050⟩⟩], []⟩ [⟨.result 593 .coefficient, true, some 1⟩, ⟨.result 2871 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 593 .coefficient)
      LeftAuthority592.bound (LeftAuthority592.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6842⟩⟩) (rawTerms := some (Proof.Events002.exact593RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority592.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority592.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 2871 .coefficient)
      LeftAuthority2870.bound (LeftAuthority2870.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨35050⟩⟩) (rawTerms := some (Proof.Events011.exact2871RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2870.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2870.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority592.bound [LeftAuthority2870.bound]
def bound : CoeffClass := .finite ⟨228855378262257504357600, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority592.bound, LeftAuthority2870.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority592.actual selector witness) * ([LeftAuthority2870.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61178

namespace LeftBound61179
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61179
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61177, .transfer 61178]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61177)
      LeftBound61177.bound (LeftBound61177.actual selector witness) := by
  exact .transfer (LeftBound61177.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61178)
      LeftBound61178.bound (LeftBound61178.actual selector witness) := by
  exact .transfer (LeftBound61178.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61177.bound, LeftBound61178.bound]
def bound : CoeffClass := .finite ⟨5601427277602316147027040, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61177.bound, LeftBound61178.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61177.actual selector witness, LeftBound61178.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61179

namespace LeftBound61180
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61180
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6857⟩⟩, ⟨.program ⟨257⟩, ⟨29393⟩⟩], []⟩ [⟨.result 603 .coefficient, true, some 1⟩, ⟨.result 2879 .coefficient, true, some 1⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 603 .coefficient)
      LeftAuthority602.bound (LeftAuthority602.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨6857⟩⟩) (rawTerms := some (Proof.Events002.exact603RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority602.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority602.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 2879 .coefficient)
      LeftAuthority2878.bound (LeftAuthority2878.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨29393⟩⟩) (rawTerms := some (Proof.Events011.exact2879RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2878.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2878.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority602.bound [LeftAuthority2878.bound]
def bound : CoeffClass := .finite ⟨228236850212900051643120, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority602.bound, LeftAuthority2878.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority602.actual selector witness) * ([LeftAuthority2878.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61180

namespace LeftBound61181
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61181
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61179, .transfer 61180]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61179)
      LeftBound61179.bound (LeftBound61179.actual selector witness) := by
  exact .transfer (LeftBound61179.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61180)
      LeftBound61180.bound (LeftBound61180.actual selector witness) := by
  exact .transfer (LeftBound61180.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61179.bound, LeftBound61180.bound]
def bound : CoeffClass := .finite ⟨5829664127815216198670160, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61179.bound, LeftBound61180.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61179.actual selector witness, LeftBound61180.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61181

namespace LeftBound61182
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61182
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6860⟩⟩, ⟨.program ⟨257⟩, ⟨26713⟩⟩], []⟩ [⟨.result 613 .coefficient, true, some 1⟩, ⟨.result 2887 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2887 .coefficient)
      LeftAuthority2886.bound (LeftAuthority2886.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨26713⟩⟩) (rawTerms := some (Proof.Events011.exact2887RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2886.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2886.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority612.bound [LeftAuthority2886.bound]
def bound : CoeffClass := .finite ⟨227009770373045750290200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority612.bound, LeftAuthority2886.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority612.actual selector witness) * ([LeftAuthority2886.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61182

namespace LeftBound61183
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61183
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61181, .transfer 61182]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61181)
      LeftBound61181.bound (LeftBound61181.actual selector witness) := by
  exact .transfer (LeftBound61181.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61182)
      LeftBound61182.bound (LeftBound61182.actual selector witness) := by
  exact .transfer (LeftBound61182.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61181.bound, LeftBound61182.bound]
def bound : CoeffClass := .finite ⟨6056673898188261948960360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61181.bound, LeftBound61182.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61181.actual selector witness, LeftBound61182.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61183

namespace LeftBound61184
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61184
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6870⟩⟩, ⟨.program ⟨257⟩, ⟨67078⟩⟩], []⟩ [⟨.result 623 .coefficient, true, some 1⟩, ⟨.result 2895 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2895 .coefficient)
      LeftAuthority2894.bound (LeftAuthority2894.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨67078⟩⟩) (rawTerms := some (Proof.Events011.exact2895RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2894.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2894.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority622.bound [LeftAuthority2894.bound]
def bound : CoeffClass := .finite ⟨226487908831958288795280, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority622.bound, LeftAuthority2894.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority622.actual selector witness) * ([LeftAuthority2894.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61184

namespace LeftBound61185
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61185
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61183, .transfer 61184]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61183)
      LeftBound61183.bound (LeftBound61183.actual selector witness) := by
  exact .transfer (LeftBound61183.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61184)
      LeftBound61184.bound (LeftBound61184.actual selector witness) := by
  exact .transfer (LeftBound61184.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61183.bound, LeftBound61184.bound]
def bound : CoeffClass := .finite ⟨6283161807020220237755640, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61183.bound, LeftBound61184.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61183.actual selector witness, LeftBound61184.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61185

namespace LeftBound61186
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61186
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6732⟩⟩, ⟨.program ⟨257⟩, ⟨63218⟩⟩], []⟩ [⟨.result 633 .coefficient, true, some 1⟩, ⟨.result 2903 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2903 .coefficient)
      LeftAuthority2902.bound (LeftAuthority2902.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨63218⟩⟩) (rawTerms := some (Proof.Events011.exact2903RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2902.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority632.bound [LeftAuthority2902.bound]
def bound : CoeffClass := .finite ⟨224377773035387248837560, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority632.bound, LeftAuthority2902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority632.actual selector witness) * ([LeftAuthority2902.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61186

namespace LeftBound61187
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61187
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61185, .transfer 61186]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61185)
      LeftBound61185.bound (LeftBound61185.actual selector witness) := by
  exact .transfer (LeftBound61185.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61186)
      LeftBound61186.bound (LeftBound61186.actual selector witness) := by
  exact .transfer (LeftBound61186.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61185.bound, LeftBound61186.bound]
def bound : CoeffClass := .finite ⟨6507539580055607486593200, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61185.bound, LeftBound61186.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61185.actual selector witness, LeftBound61186.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61187

namespace LeftBound61188
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61188
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6736⟩⟩, ⟨.program ⟨257⟩, ⟨60238⟩⟩], []⟩ [⟨.result 643 .coefficient, true, some 1⟩, ⟨.result 2911 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2911 .coefficient)
      LeftAuthority2910.bound (LeftAuthority2910.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨60238⟩⟩) (rawTerms := some (Proof.Events011.exact2911RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2910.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2910.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority642.bound [LeftAuthority2910.bound]
def bound : CoeffClass := .finite ⟨222230617312560576599880, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority642.bound, LeftAuthority2910.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority642.actual selector witness) * ([LeftAuthority2910.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61188

namespace LeftBound61189
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61189
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61187, .transfer 61188]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61187)
      LeftBound61187.bound (LeftBound61187.actual selector witness) := by
  exact .transfer (LeftBound61187.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61188)
      LeftBound61188.bound (LeftBound61188.actual selector witness) := by
  exact .transfer (LeftBound61188.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61187.bound, LeftBound61188.bound]
def bound : CoeffClass := .finite ⟨6729770197368168063193080, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61187.bound, LeftBound61188.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61187.actual selector witness, LeftBound61188.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61189

namespace LeftBound61190
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61190
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6741⟩⟩, ⟨.program ⟨257⟩, ⟨57258⟩⟩], []⟩ [⟨.result 653 .coefficient, true, some 1⟩, ⟨.result 2919 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2919 .coefficient)
      LeftAuthority2918.bound (LeftAuthority2918.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨57258⟩⟩) (rawTerms := some (Proof.Events011.exact2919RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2918.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2918.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority652.bound [LeftAuthority2918.bound]
def bound : CoeffClass := .finite ⟨220778129617707239497920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority652.bound, LeftAuthority2918.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority652.actual selector witness) * ([LeftAuthority2918.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61190

namespace LeftBound61191
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61191
def frameStart : Nat := 0
def rule : BoundRule := .sum [.transfer 61189, .transfer 61190]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61189)
      LeftBound61189.bound (LeftBound61189.actual selector witness) := by
  exact .transfer (LeftBound61189.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 61190)
      LeftBound61190.bound (LeftBound61190.actual selector witness) := by
  exact .transfer (LeftBound61190.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound61189.bound, LeftBound61190.bound]
def bound : CoeffClass := .finite ⟨6950548326985875302691000, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound61189.bound, LeftBound61190.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound61189.actual selector witness, LeftBound61190.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound61191

namespace LeftBound61192
def owner : Owner := ⟨.program ⟨257⟩, ⟨67612⟩⟩
def transferEvent : Nat := 61192
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[⟨.program ⟨257⟩, ⟨6757⟩⟩, ⟨.program ⟨257⟩, ⟨54278⟩⟩], []⟩ [⟨.result 663 .coefficient, true, some 1⟩, ⟨.result 2927 .coefficient, true, some 1⟩]
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
    BoundInputAt history owner (.result 2927 .coefficient)
      LeftAuthority2926.bound (LeftAuthority2926.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨54278⟩⟩) (rawTerms := some (Proof.Events011.exact2927RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2926.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority662.bound [LeftAuthority2926.bound]
def bound : CoeffClass := .finite ⟨216532396355828254122960, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority662.bound, LeftAuthority2926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority662.actual selector witness) * ([LeftAuthority2926.actual selector witness].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.cons (.intro (input1 selector witness)) (.nil))
end LeftBound61192

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
