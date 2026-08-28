import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard2064

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound304093
def owner : Owner := ⟨.program ⟨257⟩, ⟨31917⟩⟩
def transferEvent : Nat := 304093
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304091 .coefficient, .predecessor 1 304092 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304091 .coefficient)
      LeftBound304089.bound (LeftBound304089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304089.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304089.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304092 .coefficient)
      LeftAuthority304012.bound (LeftAuthority304012.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304013RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority304012.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority304012.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304089.bound, LeftAuthority304012.bound]
def bound : CoeffClass := .finite ⟨197, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304089.bound, LeftAuthority304012.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304089.actual selector witness, LeftAuthority304012.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304093

namespace LeftBound304097
def owner : Owner := ⟨.program ⟨257⟩, ⟨50972⟩⟩
def transferEvent : Nat := 304097
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304095 .coefficient, .predecessor 1 304096 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304095 .coefficient)
      LeftBound304093.bound (LeftBound304093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304094RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304093.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304096 .coefficient)
      LeftAuthority303989.bound (LeftAuthority303989.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303990RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303989.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303989.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304093.bound, LeftAuthority303989.bound]
def bound : CoeffClass := .finite ⟨255, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304093.bound, LeftAuthority303989.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304093.actual selector witness, LeftAuthority303989.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304097

namespace LeftBound304101
def owner : Owner := ⟨.program ⟨257⟩, ⟨53952⟩⟩
def transferEvent : Nat := 304101
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304099 .coefficient, .predecessor 1 304100 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304099 .coefficient)
      LeftBound304097.bound (LeftBound304097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304097.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304100 .coefficient)
      LeftAuthority303966.bound (LeftAuthority303966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303966.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304097.bound, LeftAuthority303966.bound]
def bound : CoeffClass := .finite ⟨314, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304097.bound, LeftAuthority303966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304097.actual selector witness, LeftAuthority303966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304101

namespace LeftBound304105
def owner : Owner := ⟨.program ⟨257⟩, ⟨56932⟩⟩
def transferEvent : Nat := 304105
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304103 .coefficient, .predecessor 1 304104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304103 .coefficient)
      LeftBound304101.bound (LeftBound304101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304104 .coefficient)
      LeftAuthority303943.bound (LeftAuthority303943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303944RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303943.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304101.bound, LeftAuthority303943.bound]
def bound : CoeffClass := .finite ⟨374, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304101.bound, LeftAuthority303943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304101.actual selector witness, LeftAuthority303943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304105

namespace LeftBound304109
def owner : Owner := ⟨.program ⟨257⟩, ⟨59912⟩⟩
def transferEvent : Nat := 304109
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304107 .coefficient, .predecessor 1 304108 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304107 .coefficient)
      LeftBound304105.bound (LeftBound304105.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304106RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304105.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304105.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304108 .coefficient)
      LeftAuthority303920.bound (LeftAuthority303920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304105.bound, LeftAuthority303920.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304105.bound, LeftAuthority303920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304105.actual selector witness, LeftAuthority303920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304109

namespace LeftBound304113
def owner : Owner := ⟨.program ⟨257⟩, ⟨62892⟩⟩
def transferEvent : Nat := 304113
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304111 .coefficient, .predecessor 1 304112 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304111 .coefficient)
      LeftBound304109.bound (LeftBound304109.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304110RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304109.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304109.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304112 .coefficient)
      LeftAuthority303897.bound (LeftAuthority303897.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303898RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303897.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303897.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304109.bound, LeftAuthority303897.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304109.bound, LeftAuthority303897.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304109.actual selector witness, LeftAuthority303897.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304113

namespace LeftBound304117
def owner : Owner := ⟨.program ⟨257⟩, ⟨65902⟩⟩
def transferEvent : Nat := 304117
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304115 .coefficient, .predecessor 1 304116 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304115 .coefficient)
      LeftBound304113.bound (LeftBound304113.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304114RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304113.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304113.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304116 .coefficient)
      LeftAuthority303874.bound (LeftAuthority303874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact303875RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303874.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304113.bound, LeftAuthority303874.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304113.bound, LeftAuthority303874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304113.actual selector witness, LeftAuthority303874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304117

namespace LeftBound304121
def owner : Owner := ⟨.program ⟨257⟩, ⟨65903⟩⟩
def transferEvent : Nat := 304121
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304119 .coefficient, .predecessor 1 304120 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304119 .coefficient)
      LeftBound304117.bound (LeftBound304117.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304118RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304117.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304117.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304120 .coefficient)
      LeftAuthority303851.bound (LeftAuthority303851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303851.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304117.bound, LeftAuthority303851.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304117.bound, LeftAuthority303851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304117.actual selector witness, LeftAuthority303851.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304121

namespace LeftBound304125
def owner : Owner := ⟨.program ⟨257⟩, ⟨65904⟩⟩
def transferEvent : Nat := 304125
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304123 .coefficient, .predecessor 1 304124 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304123 .coefficient)
      LeftBound304121.bound (LeftBound304121.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304122RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304121.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304121.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304124 .coefficient)
      LeftAuthority303828.bound (LeftAuthority303828.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303829RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303828.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303828.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304121.bound, LeftAuthority303828.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304121.bound, LeftAuthority303828.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304121.actual selector witness, LeftAuthority303828.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304125

namespace LeftBound304129
def owner : Owner := ⟨.program ⟨257⟩, ⟨65905⟩⟩
def transferEvent : Nat := 304129
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304127 .coefficient, .predecessor 1 304128 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304127 .coefficient)
      LeftBound304125.bound (LeftBound304125.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1187.exact304126RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304125.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304125.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304128 .coefficient)
      LeftAuthority303805.bound (LeftAuthority303805.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303806RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303805.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303805.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304125.bound, LeftAuthority303805.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304125.bound, LeftAuthority303805.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304125.actual selector witness, LeftAuthority303805.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304129

namespace LeftBound304133
def owner : Owner := ⟨.program ⟨257⟩, ⟨65906⟩⟩
def transferEvent : Nat := 304133
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304131 .coefficient, .predecessor 1 304132 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304131 .coefficient)
      LeftBound304129.bound (LeftBound304129.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304130RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304129.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304129.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304132 .coefficient)
      LeftAuthority303782.bound (LeftAuthority303782.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303783RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303782.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303782.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304129.bound, LeftAuthority303782.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304129.bound, LeftAuthority303782.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304129.actual selector witness, LeftAuthority303782.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304133

namespace LeftBound304137
def owner : Owner := ⟨.program ⟨257⟩, ⟨65907⟩⟩
def transferEvent : Nat := 304137
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304135 .coefficient, .predecessor 1 304136 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304135 .coefficient)
      LeftBound304133.bound (LeftBound304133.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304134RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304133.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304133.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304136 .coefficient)
      LeftAuthority303759.bound (LeftAuthority303759.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303760RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303759.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303759.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304133.bound, LeftAuthority303759.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304133.bound, LeftAuthority303759.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304133.actual selector witness, LeftAuthority303759.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304137

namespace LeftBound304141
def owner : Owner := ⟨.program ⟨257⟩, ⟨65908⟩⟩
def transferEvent : Nat := 304141
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304139 .coefficient, .predecessor 1 304140 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304139 .coefficient)
      LeftBound304137.bound (LeftBound304137.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304138RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304137.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304137.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304140 .coefficient)
      LeftAuthority303736.bound (LeftAuthority303736.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303737RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303736.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303736.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304137.bound, LeftAuthority303736.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304137.bound, LeftAuthority303736.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304137.actual selector witness, LeftAuthority303736.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304141

namespace LeftBound304145
def owner : Owner := ⟨.program ⟨257⟩, ⟨65909⟩⟩
def transferEvent : Nat := 304145
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304143 .coefficient, .predecessor 1 304144 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304143 .coefficient)
      LeftBound304141.bound (LeftBound304141.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304142RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304141.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304141.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304144 .coefficient)
      LeftAuthority303713.bound (LeftAuthority303713.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303714RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303713.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303713.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304141.bound, LeftAuthority303713.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304141.bound, LeftAuthority303713.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304141.actual selector witness, LeftAuthority303713.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304145

namespace LeftBound304149
def owner : Owner := ⟨.program ⟨257⟩, ⟨65910⟩⟩
def transferEvent : Nat := 304149
def frameStart : Nat := 303660
def rule : BoundRule := .sum [.predecessor 0 304147 .coefficient, .predecessor 1 304148 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304147 .coefficient)
      LeftBound304145.bound (LeftBound304145.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304146RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304145.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304145.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 304148 .coefficient)
      LeftAuthority303690.bound (LeftAuthority303690.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1186.exact303691RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority303690.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority303690.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound304145.bound, LeftAuthority303690.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304145.bound, LeftAuthority303690.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound304145.actual selector witness, LeftAuthority303690.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound304149

namespace LeftBound304152
def owner : Owner := ⟨.program ⟨257⟩, ⟨65911⟩⟩
def transferEvent : Nat := 304152
def frameStart : Nat := 303660
def rule : BoundRule := .identity (.predecessor 0 304151 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 304151 .coefficient)
      LeftBound304149.bound (LeftBound304149.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events1188.exact304150RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound304149.bound, RecordedBoundRefines] <;> decide)
      (LeftBound304149.derived selector witness)

def rawBound : CoeffClass := LeftBound304149.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound304149.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound304149.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound304152

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
