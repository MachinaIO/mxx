import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard039
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard040

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound12739
def owner : Owner := ⟨.program ⟨257⟩, ⟨66241⟩⟩
def transferEvent : Nat := 12739
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12737 .coefficient, .predecessor 1 12738 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12737 .coefficient)
      LeftBound12735.bound (LeftBound12735.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12736RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12735.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12735.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12738 .coefficient)
      LeftBound12614.bound (LeftBound12614.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12616RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12614.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12614.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12735.bound, LeftBound12614.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12735.bound, LeftBound12614.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12735.actual selector witness, LeftBound12614.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12739

namespace LeftBound12743
def owner : Owner := ⟨.program ⟨257⟩, ⟨66242⟩⟩
def transferEvent : Nat := 12743
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12741 .coefficient, .predecessor 1 12742 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12741 .coefficient)
      LeftBound12739.bound (LeftBound12739.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12740RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12739.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12739.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12742 .coefficient)
      LeftBound12606.bound (LeftBound12606.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12608RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12606.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12606.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12739.bound, LeftBound12606.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12739.bound, LeftBound12606.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12739.actual selector witness, LeftBound12606.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12743

namespace LeftBound12747
def owner : Owner := ⟨.program ⟨257⟩, ⟨66243⟩⟩
def transferEvent : Nat := 12747
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12745 .coefficient, .predecessor 1 12746 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12745 .coefficient)
      LeftBound12743.bound (LeftBound12743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12746 .coefficient)
      LeftBound12598.bound (LeftBound12598.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12600RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12598.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12598.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12743.bound, LeftBound12598.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12743.bound, LeftBound12598.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12743.actual selector witness, LeftBound12598.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12747

namespace LeftBound12751
def owner : Owner := ⟨.program ⟨257⟩, ⟨66244⟩⟩
def transferEvent : Nat := 12751
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12749 .coefficient, .predecessor 1 12750 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12749 .coefficient)
      LeftBound12747.bound (LeftBound12747.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12748RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12747.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12747.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12750 .coefficient)
      LeftBound12590.bound (LeftBound12590.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12592RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12590.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12590.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12747.bound, LeftBound12590.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12747.bound, LeftBound12590.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12747.actual selector witness, LeftBound12590.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12751

namespace LeftBound12755
def owner : Owner := ⟨.program ⟨257⟩, ⟨66245⟩⟩
def transferEvent : Nat := 12755
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12753 .coefficient, .predecessor 1 12754 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12753 .coefficient)
      LeftBound12751.bound (LeftBound12751.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12752RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12751.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12751.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12754 .coefficient)
      LeftBound12582.bound (LeftBound12582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12584RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12582.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12582.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12751.bound, LeftBound12582.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12751.bound, LeftBound12582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12751.actual selector witness, LeftBound12582.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12755

namespace LeftBound12759
def owner : Owner := ⟨.program ⟨257⟩, ⟨66246⟩⟩
def transferEvent : Nat := 12759
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12757 .coefficient, .predecessor 1 12758 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12757 .coefficient)
      LeftBound12755.bound (LeftBound12755.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12756RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12755.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12755.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12758 .coefficient)
      LeftBound12574.bound (LeftBound12574.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12576RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12574.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12574.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12755.bound, LeftBound12574.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12755.bound, LeftBound12574.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12755.actual selector witness, LeftBound12574.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12759

namespace LeftBound12763
def owner : Owner := ⟨.program ⟨257⟩, ⟨66247⟩⟩
def transferEvent : Nat := 12763
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12761 .coefficient, .predecessor 1 12762 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12761 .coefficient)
      LeftBound12759.bound (LeftBound12759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12759.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12759.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12762 .coefficient)
      LeftBound12566.bound (LeftBound12566.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12568RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12566.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12566.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12759.bound, LeftBound12566.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12759.bound, LeftBound12566.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12759.actual selector witness, LeftBound12566.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12763

namespace LeftBound12767
def owner : Owner := ⟨.program ⟨257⟩, ⟨66248⟩⟩
def transferEvent : Nat := 12767
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12765 .coefficient, .predecessor 1 12766 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12765 .coefficient)
      LeftBound12763.bound (LeftBound12763.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12764RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12763.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12763.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12766 .coefficient)
      LeftBound12558.bound (LeftBound12558.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12560RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12558.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12558.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12763.bound, LeftBound12558.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12763.bound, LeftBound12558.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12763.actual selector witness, LeftBound12558.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12767

namespace LeftBound12771
def owner : Owner := ⟨.program ⟨257⟩, ⟨67366⟩⟩
def transferEvent : Nat := 12771
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 12769 .coefficient, .predecessor 1 12770 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12769 .coefficient)
      LeftBound12767.bound (LeftBound12767.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12768RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12767.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12767.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12770 .coefficient)
      LeftBound12550.bound (LeftBound12550.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12552RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12550.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12550.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound12767.bound, LeftBound12550.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12767.bound, LeftBound12550.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound12767.actual selector witness, LeftBound12550.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound12771

namespace LeftBound12775
def owner : Owner := ⟨.program ⟨257⟩, ⟨67367⟩⟩
def transferEvent : Nat := 12775
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 12773 .coefficient) (.predecessor 1 12774 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 12773 .coefficient)
      LeftBound12771.bound (LeftBound12771.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events049.exact12772RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound12771.bound, RecordedBoundRefines] <;> decide)
      (LeftBound12771.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 12774 .coefficient)
      LeftAuthority12048.bound (LeftAuthority12048.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events047.exact12049RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority12048.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority12048.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound12771.bound LeftAuthority12048.bound
def bound : CoeffClass := .finite ⟨337289362800481729748408361040283865183438135403577441169979325019767313588395337813721950057960171748571934653547853057558593831647485134712044184479191622963651713758379690351440175765909625274368, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound12771.bound, LeftAuthority12048.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound12771.actual selector witness) * (LeftAuthority12048.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound12775

namespace LeftBound13298
def owner : Owner := ⟨.program ⟨257⟩, ⟨67301⟩⟩
def transferEvent : Nat := 13298
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13296 .coefficient) (.predecessor 1 13297 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13296 .coefficient)
      LeftAuthority13294.bound (LeftAuthority13294.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13295RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13294.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13294.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13297 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13294.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13294.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13294.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13298

namespace LeftBound13306
def owner : Owner := ⟨.program ⟨257⟩, ⟨48253⟩⟩
def transferEvent : Nat := 13306
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13304 .coefficient) (.predecessor 1 13305 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13304 .coefficient)
      LeftAuthority13302.bound (LeftAuthority13302.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13303RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13302.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13302.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13305 .coefficient)
      LeftAuthority542.bound (LeftAuthority542.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact543RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority542.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority542.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13302.bound LeftAuthority542.bound
def bound : CoeffClass := .finite ⟨230731242018505516688400, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13302.bound, LeftAuthority542.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13302.actual selector witness) * (LeftAuthority542.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13306

namespace LeftBound13314
def owner : Owner := ⟨.program ⟨257⟩, ⟨45573⟩⟩
def transferEvent : Nat := 13314
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13312 .coefficient) (.predecessor 1 13313 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13312 .coefficient)
      LeftAuthority13310.bound (LeftAuthority13310.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events051.exact13311RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13310.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13310.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13313 .coefficient)
      LeftAuthority552.bound (LeftAuthority552.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact553RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority552.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority552.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13310.bound LeftAuthority552.bound
def bound : CoeffClass := .finite ⟨230600885384596756509480, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13310.bound, LeftAuthority552.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13310.actual selector witness) * (LeftAuthority552.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13314

namespace LeftBound13322
def owner : Owner := ⟨.program ⟨257⟩, ⟨42896⟩⟩
def transferEvent : Nat := 13322
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13320 .coefficient) (.predecessor 1 13321 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13320 .coefficient)
      LeftAuthority13318.bound (LeftAuthority13318.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13319RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13318.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13318.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13321 .coefficient)
      LeftAuthority562.bound (LeftAuthority562.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact563RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority562.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority562.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13318.bound LeftAuthority562.bound
def bound : CoeffClass := .finite ⟨230150786063741980797360, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13318.bound, LeftAuthority562.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13318.actual selector witness) * (LeftAuthority562.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13322

namespace LeftBound13330
def owner : Owner := ⟨.program ⟨257⟩, ⟨40216⟩⟩
def transferEvent : Nat := 13330
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13328 .coefficient) (.predecessor 1 13329 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13328 .coefficient)
      LeftAuthority13326.bound (LeftAuthority13326.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13327RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13326.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13326.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13329 .coefficient)
      LeftAuthority572.bound (LeftAuthority572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority572.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13326.bound LeftAuthority572.bound
def bound : CoeffClass := .finite ⟨229585767767349815541720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13326.bound, LeftAuthority572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13326.actual selector witness) * (LeftAuthority572.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13330

namespace LeftBound13338
def owner : Owner := ⟨.program ⟨257⟩, ⟨37533⟩⟩
def transferEvent : Nat := 13338
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 13336 .coefficient) (.predecessor 1 13337 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 13336 .coefficient)
      LeftAuthority13334.bound (LeftAuthority13334.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events052.exact13335RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority13334.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority13334.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 13337 .coefficient)
      LeftAuthority582.bound (LeftAuthority582.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events002.exact583RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority582.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority582.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority13334.bound LeftAuthority582.bound
def bound : CoeffClass := .finite ⟨229121489167213617734760, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority13334.bound, LeftAuthority582.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority13334.actual selector witness) * (LeftAuthority582.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound13338

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
