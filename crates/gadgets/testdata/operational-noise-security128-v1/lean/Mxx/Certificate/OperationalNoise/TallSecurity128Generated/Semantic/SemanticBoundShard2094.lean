import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard062
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1793
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1895
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1996
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2067
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2069
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2070
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2091
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2093

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound308007
def owner : Owner := ⟨.program ⟨257⟩, ⟨69384⟩⟩
def transferEvent : Nat := 308007
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308003 .summary, .result 304888 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308003 .summary)
      LeftBound308002.bound (LeftBound308002.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69383⟩⟩) (rawTerms := some (Proof.Events1203.exact308003RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 304888 .summary)
      LeftBound304883.bound (LeftBound304883.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨47097⟩⟩) (rawTerms := some (Proof.Events1190.exact304888RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound304883.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308002.bound, LeftBound304883.bound]
def bound : CoeffClass := .finite ⟨5876032038633885316753225624840917630320692, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308002.bound, LeftBound304883.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308002.actual selector witness, LeftBound304883.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308007

namespace LeftBound308011
def owner : Owner := ⟨.program ⟨257⟩, ⟨69385⟩⟩
def transferEvent : Nat := 308011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308009 .coefficient, .predecessor 1 308010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308009 .coefficient)
      LeftBound308006.bound (LeftBound308006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308010 .coefficient)
      LeftBound304693.bound (LeftBound304693.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1190.exact304700RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304693.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304693.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308006.bound, LeftBound304693.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308006.bound, LeftBound304693.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308006.actual selector witness, LeftBound304693.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308011

namespace LeftBound308012
def owner : Owner := ⟨.program ⟨257⟩, ⟨69385⟩⟩
def transferEvent : Nat := 308012
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308008 .summary, .result 304700 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308008 .summary)
      LeftBound308007.bound (LeftBound308007.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69384⟩⟩) (rawTerms := some (Proof.Events1203.exact308008RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 304700 .summary)
      LeftBound304695.bound (LeftBound304695.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨49777⟩⟩) (rawTerms := some (Proof.Events1190.exact304700RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound304695.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308007.bound, LeftBound304695.bound]
def bound : CoeffClass := .finite ⟨6221717896068416040249469304417135687106612, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308007.bound, LeftBound304695.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308007.actual selector witness, LeftBound304695.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308012

namespace LeftBound308016
def owner : Owner := ⟨.program ⟨257⟩, ⟨70940⟩⟩
def transferEvent : Nat := 308016
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308014 .coefficient, .predecessor 1 308015 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308014 .coefficient)
      LeftBound308011.bound (LeftBound308011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308015 .coefficient)
      LeftBound304505.bound (LeftBound304505.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1189.exact304512RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304505.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304505.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308011.bound, LeftBound304505.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308011.bound, LeftBound304505.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308011.actual selector witness, LeftBound304505.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308016

namespace LeftBound308017
def owner : Owner := ⟨.program ⟨257⟩, ⟨70940⟩⟩
def transferEvent : Nat := 308017
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308013 .summary, .result 304512 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308013 .summary)
      LeftBound308012.bound (LeftBound308012.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨69385⟩⟩) (rawTerms := some (Proof.Events1203.exact308013RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308012.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 304512 .summary)
      LeftBound304507.bound (LeftBound304507.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70938⟩⟩) (rawTerms := some (Proof.Events1189.exact304512RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound304507.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308012.bound, LeftBound304507.bound]
def bound : CoeffClass := .finite ⟨66805187227601152574551644069558752530002096506798132, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308012.bound, LeftBound304507.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308012.actual selector witness, LeftBound304507.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308017

namespace LeftBound308021
def owner : Owner := ⟨.program ⟨257⟩, ⟨70941⟩⟩
def transferEvent : Nat := 308021
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 308019 .coefficient) (.predecessor 1 308020 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308019 .coefficient)
      LeftBound308016.bound (LeftBound308016.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308018RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308016.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308016.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308020 .coefficient)
      LeftBound16733.bound (LeftBound16733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events065.exact16734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound16733.bound, RecordedBoundRefines] <;> decide)
      (LeftBound16733.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound308016.bound LeftBound16733.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308016.bound, LeftBound16733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound308016.actual selector witness) * (LeftBound16733.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound308021

namespace LeftBound308022
def owner : Owner := ⟨.program ⟨257⟩, ⟨70941⟩⟩
def transferEvent : Nat := 308022
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7109⟩⟩]⟩ [⟨.result 16730 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 16730 .coefficient)
      LeftAuthority16729.bound (LeftAuthority16729.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7109⟩⟩) (rawTerms := some (Proof.Events065.exact16730RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority16729.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority16729.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority16729.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority16729.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority16729.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound308022

namespace LeftBound308023
def owner : Owner := ⟨.program ⟨257⟩, ⟨70941⟩⟩
def transferEvent : Nat := 308023
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 308018 .summary) (.transfer 308022) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308018 .summary)
      LeftBound308017.bound (LeftBound308017.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70940⟩⟩) (rawTerms := some (Proof.Events1203.exact308018RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308017.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 308022)
      LeftBound308022.bound (LeftBound308022.actual selector witness) := by
  exact .transfer (LeftBound308022.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound308017.bound LeftBound308022.bound
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167290658270334546534727680, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308017.bound, LeftBound308022.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound308017.actual selector witness) * (LeftBound308022.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound308023

namespace LeftBound308103
def owner : Owner := ⟨.program ⟨257⟩, ⟨70942⟩⟩
def transferEvent : Nat := 308103
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308101 .coefficient, .predecessor 1 308102 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308101 .coefficient)
      LeftBound307919.bound (LeftBound307919.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1202.exact307923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound307919.bound, RecordedBoundRefines] <;> decide)
      (LeftBound307919.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308102 .coefficient)
      LeftBound308021.bound (LeftBound308021.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308100RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308021.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308021.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307919.bound, LeftBound308021.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307919.bound, LeftBound308021.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307919.actual selector witness, LeftBound308021.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308103

namespace LeftBound308104
def owner : Owner := ⟨.program ⟨257⟩, ⟨70942⟩⟩
def transferEvent : Nat := 308104
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 307923 .summary, .result 308100 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 307923 .summary)
      LeftBound307922.bound (LeftBound307922.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨9447⟩⟩) (rawTerms := some (Proof.Events1202.exact307923RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound307922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308100 .summary)
      LeftBound308023.bound (LeftBound308023.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70941⟩⟩) (rawTerms := some (Proof.Events1203.exact308100RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308023.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound307922.bound, LeftBound308023.bound]
def bound : CoeffClass := .finite ⟨717315235864259647099013782854467978167290658270334546534727732, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound307922.bound, LeftBound308023.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound307922.actual selector witness, LeftBound308023.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308104

namespace LeftBound308108
def owner : Owner := ⟨.program ⟨257⟩, ⟨71059⟩⟩
def transferEvent : Nat := 308108
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308106 .coefficient, .predecessor 1 308107 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308106 .coefficient)
      LeftBound308103.bound (LeftBound308103.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308105RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308103.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308103.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308107 .coefficient)
      LeftBound295012.bound (LeftBound295012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1152.exact295073RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound295012.bound, RecordedBoundRefines] <;> decide)
      (LeftBound295012.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308103.bound, LeftBound295012.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308103.bound, LeftBound295012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308103.actual selector witness, LeftBound295012.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308108

namespace LeftBound308109
def owner : Owner := ⟨.program ⟨257⟩, ⟨71059⟩⟩
def transferEvent : Nat := 308109
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308105 .summary, .result 295073 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308105 .summary)
      LeftBound308104.bound (LeftBound308104.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70942⟩⟩) (rawTerms := some (Proof.Events1203.exact308105RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308104.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 295073 .summary)
      LeftBound295014.bound (LeftBound295014.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71058⟩⟩) (rawTerms := some (Proof.Events1152.exact295073RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound295014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308104.bound, LeftBound295014.bound]
def bound : CoeffClass := .finite ⟨7702113698116118934721173325132051628004677497839473894998983394742763572, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308104.bound, LeftBound295014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308104.actual selector witness, LeftBound295014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308109

namespace LeftBound308113
def owner : Owner := ⟨.program ⟨257⟩, ⟨71060⟩⟩
def transferEvent : Nat := 308113
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308111 .coefficient, .predecessor 1 308112 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308111 .coefficient)
      LeftBound308108.bound (LeftBound308108.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308108.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308108.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308112 .coefficient)
      LeftBound280423.bound (LeftBound280423.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1095.exact280484RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound280423.bound, RecordedBoundRefines] <;> decide)
      (LeftBound280423.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308108.bound, LeftBound280423.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308108.bound, LeftBound280423.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308108.actual selector witness, LeftBound280423.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308113

namespace LeftBound308114
def owner : Owner := ⟨.program ⟨257⟩, ⟨71060⟩⟩
def transferEvent : Nat := 308114
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308110 .summary, .result 280484 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308110 .summary)
      LeftBound308109.bound (LeftBound308109.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71059⟩⟩) (rawTerms := some (Proof.Events1203.exact308110RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 280484 .summary)
      LeftBound280425.bound (LeftBound280425.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨70989⟩⟩) (rawTerms := some (Proof.Events1095.exact280484RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound280425.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308109.bound, LeftBound280425.bound]
def bound : CoeffClass := .finite ⟨15404227395514922633578087003165089473154887017511657131727632242950799412, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308109.bound, LeftBound280425.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308109.actual selector witness, LeftBound280425.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308114

namespace LeftBound308118
def owner : Owner := ⟨.program ⟨257⟩, ⟨71093⟩⟩
def transferEvent : Nat := 308118
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 308116 .coefficient, .predecessor 1 308117 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 308116 .coefficient)
      LeftBound308113.bound (LeftBound308113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1203.exact308115RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound308113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound308113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 308117 .coefficient)
      LeftBound265798.bound (LeftBound265798.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1038.exact265859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound265798.bound, RecordedBoundRefines] <;> decide)
      (LeftBound265798.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308113.bound, LeftBound265798.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308113.bound, LeftBound265798.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308113.actual selector witness, LeftBound265798.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308118

namespace LeftBound308119
def owner : Owner := ⟨.program ⟨257⟩, ⟨71093⟩⟩
def transferEvent : Nat := 308119
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 308115 .summary, .result 265859 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 308115 .summary)
      LeftBound308114.bound (LeftBound308114.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71060⟩⟩) (rawTerms := some (Proof.Events1203.exact308115RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound308114.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 265859 .summary)
      LeftBound265800.bound (LeftBound265800.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨71092⟩⟩) (rawTerms := some (Proof.Events1038.exact265859RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound265800.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound308114.bound, LeftBound265800.bound]
def bound : CoeffClass := .finite ⟨23106341092913726332435000681198127318305096537183840368456281091158835252, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound308114.bound, LeftBound265800.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound308114.actual selector witness, LeftBound265800.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound308119

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
