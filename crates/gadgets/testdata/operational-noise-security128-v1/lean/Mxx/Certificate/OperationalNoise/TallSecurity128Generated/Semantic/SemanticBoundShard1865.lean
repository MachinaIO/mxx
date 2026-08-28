import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1864

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound275902
def owner : Owner := ⟨.program ⟨257⟩, ⟨21930⟩⟩
def transferEvent : Nat := 275902
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275900 .coefficient, .predecessor 1 275901 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275900 .coefficient)
      LeftBound275898.bound (LeftBound275898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275899RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275898.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275901 .coefficient)
      LeftAuthority275848.bound (LeftAuthority275848.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275849RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275848.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275848.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275898.bound, LeftAuthority275848.bound]
def bound : CoeffClass := .finite ⟨142, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275898.bound, LeftAuthority275848.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275898.actual selector witness, LeftAuthority275848.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275902

namespace LeftBound275906
def owner : Owner := ⟨.program ⟨257⟩, ⟨31950⟩⟩
def transferEvent : Nat := 275906
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275904 .coefficient, .predecessor 1 275905 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275904 .coefficient)
      LeftBound275902.bound (LeftBound275902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275902.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275902.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275905 .coefficient)
      LeftAuthority275825.bound (LeftAuthority275825.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275826RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275825.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275825.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275902.bound, LeftAuthority275825.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275902.bound, LeftAuthority275825.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275902.actual selector witness, LeftAuthority275825.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275906

namespace LeftBound275910
def owner : Owner := ⟨.program ⟨257⟩, ⟨51005⟩⟩
def transferEvent : Nat := 275910
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275908 .coefficient, .predecessor 1 275909 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275908 .coefficient)
      LeftBound275906.bound (LeftBound275906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275907RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275906.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275909 .coefficient)
      LeftAuthority275802.bound (LeftAuthority275802.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275803RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275802.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275802.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275906.bound, LeftAuthority275802.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275906.bound, LeftAuthority275802.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275906.actual selector witness, LeftAuthority275802.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275910

namespace LeftBound275914
def owner : Owner := ⟨.program ⟨257⟩, ⟨53985⟩⟩
def transferEvent : Nat := 275914
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275912 .coefficient, .predecessor 1 275913 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275912 .coefficient)
      LeftBound275910.bound (LeftBound275910.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275911RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275910.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275910.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275913 .coefficient)
      LeftAuthority275779.bound (LeftAuthority275779.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275780RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275779.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275779.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275910.bound, LeftAuthority275779.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275910.bound, LeftAuthority275779.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275910.actual selector witness, LeftAuthority275779.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275914

namespace LeftBound275918
def owner : Owner := ⟨.program ⟨257⟩, ⟨56965⟩⟩
def transferEvent : Nat := 275918
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275916 .coefficient, .predecessor 1 275917 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275916 .coefficient)
      LeftBound275914.bound (LeftBound275914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275914.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275917 .coefficient)
      LeftAuthority275756.bound (LeftAuthority275756.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275757RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275756.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275756.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275914.bound, LeftAuthority275756.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275914.bound, LeftAuthority275756.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275914.actual selector witness, LeftAuthority275756.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275918

namespace LeftBound275922
def owner : Owner := ⟨.program ⟨257⟩, ⟨59945⟩⟩
def transferEvent : Nat := 275922
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275920 .coefficient, .predecessor 1 275921 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275920 .coefficient)
      LeftBound275918.bound (LeftBound275918.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275919RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275918.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275918.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275921 .coefficient)
      LeftAuthority275733.bound (LeftAuthority275733.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275734RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275733.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275733.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275918.bound, LeftAuthority275733.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275918.bound, LeftAuthority275733.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275918.actual selector witness, LeftAuthority275733.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275922

namespace LeftBound275926
def owner : Owner := ⟨.program ⟨257⟩, ⟨62925⟩⟩
def transferEvent : Nat := 275926
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275924 .coefficient, .predecessor 1 275925 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275924 .coefficient)
      LeftBound275922.bound (LeftBound275922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275923RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275922.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275925 .coefficient)
      LeftAuthority275710.bound (LeftAuthority275710.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275711RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275710.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275710.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275922.bound, LeftAuthority275710.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275922.bound, LeftAuthority275710.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275922.actual selector witness, LeftAuthority275710.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275926

namespace LeftBound275930
def owner : Owner := ⟨.program ⟨257⟩, ⟨66020⟩⟩
def transferEvent : Nat := 275930
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275928 .coefficient, .predecessor 1 275929 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275928 .coefficient)
      LeftBound275926.bound (LeftBound275926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275926.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275926.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275929 .coefficient)
      LeftAuthority275687.bound (LeftAuthority275687.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275688RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275687.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275687.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275926.bound, LeftAuthority275687.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275926.bound, LeftAuthority275687.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275926.actual selector witness, LeftAuthority275687.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275930

namespace LeftBound275934
def owner : Owner := ⟨.program ⟨257⟩, ⟨66021⟩⟩
def transferEvent : Nat := 275934
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275932 .coefficient, .predecessor 1 275933 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275932 .coefficient)
      LeftBound275930.bound (LeftBound275930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275931RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275930.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275933 .coefficient)
      LeftAuthority275664.bound (LeftAuthority275664.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275665RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275664.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275664.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275930.bound, LeftAuthority275664.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275930.bound, LeftAuthority275664.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275930.actual selector witness, LeftAuthority275664.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275934

namespace LeftBound275938
def owner : Owner := ⟨.program ⟨257⟩, ⟨66022⟩⟩
def transferEvent : Nat := 275938
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275936 .coefficient, .predecessor 1 275937 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275936 .coefficient)
      LeftBound275934.bound (LeftBound275934.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275935RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275934.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275934.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275937 .coefficient)
      LeftAuthority275641.bound (LeftAuthority275641.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275642RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275641.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275641.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275934.bound, LeftAuthority275641.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275934.bound, LeftAuthority275641.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275934.actual selector witness, LeftAuthority275641.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275938

namespace LeftBound275942
def owner : Owner := ⟨.program ⟨257⟩, ⟨66023⟩⟩
def transferEvent : Nat := 275942
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275940 .coefficient, .predecessor 1 275941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275940 .coefficient)
      LeftBound275938.bound (LeftBound275938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275938.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275941 .coefficient)
      LeftAuthority275618.bound (LeftAuthority275618.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275619RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275618.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275618.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275938.bound, LeftAuthority275618.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275938.bound, LeftAuthority275618.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275938.actual selector witness, LeftAuthority275618.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275942

namespace LeftBound275946
def owner : Owner := ⟨.program ⟨257⟩, ⟨66024⟩⟩
def transferEvent : Nat := 275946
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275944 .coefficient, .predecessor 1 275945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275944 .coefficient)
      LeftBound275942.bound (LeftBound275942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275945 .coefficient)
      LeftAuthority275595.bound (LeftAuthority275595.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275596RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275595.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275595.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275942.bound, LeftAuthority275595.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275942.bound, LeftAuthority275595.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275942.actual selector witness, LeftAuthority275595.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275946

namespace LeftBound275950
def owner : Owner := ⟨.program ⟨257⟩, ⟨66025⟩⟩
def transferEvent : Nat := 275950
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275948 .coefficient, .predecessor 1 275949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275948 .coefficient)
      LeftBound275946.bound (LeftBound275946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275949 .coefficient)
      LeftAuthority275572.bound (LeftAuthority275572.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275573RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275572.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275572.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275946.bound, LeftAuthority275572.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275946.bound, LeftAuthority275572.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275946.actual selector witness, LeftAuthority275572.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275950

namespace LeftBound275954
def owner : Owner := ⟨.program ⟨257⟩, ⟨66026⟩⟩
def transferEvent : Nat := 275954
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275952 .coefficient, .predecessor 1 275953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275952 .coefficient)
      LeftBound275950.bound (LeftBound275950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275953 .coefficient)
      LeftAuthority275549.bound (LeftAuthority275549.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275550RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275549.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275549.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275950.bound, LeftAuthority275549.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275950.bound, LeftAuthority275549.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275950.actual selector witness, LeftAuthority275549.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275954

namespace LeftBound275958
def owner : Owner := ⟨.program ⟨257⟩, ⟨66027⟩⟩
def transferEvent : Nat := 275958
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275956 .coefficient, .predecessor 1 275957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275956 .coefficient)
      LeftBound275954.bound (LeftBound275954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275957 .coefficient)
      LeftAuthority275526.bound (LeftAuthority275526.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275527RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275526.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275526.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275954.bound, LeftAuthority275526.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275954.bound, LeftAuthority275526.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275954.actual selector witness, LeftAuthority275526.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275958

namespace LeftBound275962
def owner : Owner := ⟨.program ⟨257⟩, ⟨66028⟩⟩
def transferEvent : Nat := 275962
def frameStart : Nat := 275461
def rule : BoundRule := .sum [.predecessor 0 275960 .coefficient, .predecessor 1 275961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 275960 .coefficient)
      LeftBound275958.bound (LeftBound275958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1077.exact275959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound275958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound275958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 275961 .coefficient)
      LeftAuthority275503.bound (LeftAuthority275503.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1076.exact275504RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority275503.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority275503.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound275958.bound, LeftAuthority275503.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound275958.bound, LeftAuthority275503.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound275958.actual selector witness, LeftAuthority275503.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound275962

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
