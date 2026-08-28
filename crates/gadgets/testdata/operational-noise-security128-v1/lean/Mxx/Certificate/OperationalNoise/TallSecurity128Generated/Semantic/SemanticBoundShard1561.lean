import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1560

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound232047
def owner : Owner := ⟨.program ⟨257⟩, ⟨60083⟩⟩
def transferEvent : Nat := 232047
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232045 .coefficient, .predecessor 1 232046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232045 .coefficient)
      LeftBound232043.bound (LeftBound232043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232046 .coefficient)
      LeftAuthority231858.bound (LeftAuthority231858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231859RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231858.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232043.bound, LeftAuthority231858.bound]
def bound : CoeffClass := .finite ⟨435, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232043.bound, LeftAuthority231858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232043.actual selector witness, LeftAuthority231858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232047

namespace LeftBound232051
def owner : Owner := ⟨.program ⟨257⟩, ⟨63063⟩⟩
def transferEvent : Nat := 232051
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232049 .coefficient, .predecessor 1 232050 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232049 .coefficient)
      LeftBound232047.bound (LeftBound232047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232050 .coefficient)
      LeftAuthority231835.bound (LeftAuthority231835.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231835.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231835.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232047.bound, LeftAuthority231835.bound]
def bound : CoeffClass := .finite ⟨496, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232047.bound, LeftAuthority231835.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232047.actual selector witness, LeftAuthority231835.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232051

namespace LeftBound232055
def owner : Owner := ⟨.program ⟨257⟩, ⟨66532⟩⟩
def transferEvent : Nat := 232055
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232053 .coefficient, .predecessor 1 232054 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232053 .coefficient)
      LeftBound232051.bound (LeftBound232051.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232052RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232051.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232051.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232054 .coefficient)
      LeftAuthority231812.bound (LeftAuthority231812.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231813RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231812.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231812.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232051.bound, LeftAuthority231812.bound]
def bound : CoeffClass := .finite ⟨558, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232051.bound, LeftAuthority231812.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232051.actual selector witness, LeftAuthority231812.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232055

namespace LeftBound232059
def owner : Owner := ⟨.program ⟨257⟩, ⟨66533⟩⟩
def transferEvent : Nat := 232059
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232057 .coefficient, .predecessor 1 232058 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232057 .coefficient)
      LeftBound232055.bound (LeftBound232055.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232056RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232055.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232055.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232058 .coefficient)
      LeftAuthority231789.bound (LeftAuthority231789.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231790RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231789.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231789.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232055.bound, LeftAuthority231789.bound]
def bound : CoeffClass := .finite ⟨620, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232055.bound, LeftAuthority231789.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232055.actual selector witness, LeftAuthority231789.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232059

namespace LeftBound232063
def owner : Owner := ⟨.program ⟨257⟩, ⟨66534⟩⟩
def transferEvent : Nat := 232063
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232061 .coefficient, .predecessor 1 232062 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232061 .coefficient)
      LeftBound232059.bound (LeftBound232059.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232060RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232059.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232059.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232062 .coefficient)
      LeftAuthority231766.bound (LeftAuthority231766.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231767RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231766.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231766.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232059.bound, LeftAuthority231766.bound]
def bound : CoeffClass := .finite ⟨682, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232059.bound, LeftAuthority231766.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232059.actual selector witness, LeftAuthority231766.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232063

namespace LeftBound232067
def owner : Owner := ⟨.program ⟨257⟩, ⟨66535⟩⟩
def transferEvent : Nat := 232067
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232065 .coefficient, .predecessor 1 232066 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232065 .coefficient)
      LeftBound232063.bound (LeftBound232063.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232064RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232063.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232063.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232066 .coefficient)
      LeftAuthority231743.bound (LeftAuthority231743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231744RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231743.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231743.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232063.bound, LeftAuthority231743.bound]
def bound : CoeffClass := .finite ⟨744, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232063.bound, LeftAuthority231743.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232063.actual selector witness, LeftAuthority231743.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232067

namespace LeftBound232071
def owner : Owner := ⟨.program ⟨257⟩, ⟨66536⟩⟩
def transferEvent : Nat := 232071
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232069 .coefficient, .predecessor 1 232070 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232069 .coefficient)
      LeftBound232067.bound (LeftBound232067.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232068RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232067.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232067.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232070 .coefficient)
      LeftAuthority231720.bound (LeftAuthority231720.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231721RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231720.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231720.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232067.bound, LeftAuthority231720.bound]
def bound : CoeffClass := .finite ⟨807, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232067.bound, LeftAuthority231720.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232067.actual selector witness, LeftAuthority231720.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232071

namespace LeftBound232075
def owner : Owner := ⟨.program ⟨257⟩, ⟨66537⟩⟩
def transferEvent : Nat := 232075
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232073 .coefficient, .predecessor 1 232074 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232073 .coefficient)
      LeftBound232071.bound (LeftBound232071.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232072RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232071.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232071.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232074 .coefficient)
      LeftAuthority231697.bound (LeftAuthority231697.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events905.exact231698RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231697.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231697.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232071.bound, LeftAuthority231697.bound]
def bound : CoeffClass := .finite ⟨870, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232071.bound, LeftAuthority231697.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232071.actual selector witness, LeftAuthority231697.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232075

namespace LeftBound232079
def owner : Owner := ⟨.program ⟨257⟩, ⟨66538⟩⟩
def transferEvent : Nat := 232079
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232077 .coefficient, .predecessor 1 232078 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232077 .coefficient)
      LeftBound232075.bound (LeftBound232075.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232076RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232075.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232075.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232078 .coefficient)
      LeftAuthority231674.bound (LeftAuthority231674.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events904.exact231675RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231674.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231674.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232075.bound, LeftAuthority231674.bound]
def bound : CoeffClass := .finite ⟨933, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232075.bound, LeftAuthority231674.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232075.actual selector witness, LeftAuthority231674.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232079

namespace LeftBound232083
def owner : Owner := ⟨.program ⟨257⟩, ⟨66539⟩⟩
def transferEvent : Nat := 232083
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232081 .coefficient, .predecessor 1 232082 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232081 .coefficient)
      LeftBound232079.bound (LeftBound232079.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232080RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232079.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232079.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232082 .coefficient)
      LeftAuthority231651.bound (LeftAuthority231651.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events904.exact231652RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231651.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231651.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232079.bound, LeftAuthority231651.bound]
def bound : CoeffClass := .finite ⟨996, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232079.bound, LeftAuthority231651.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232079.actual selector witness, LeftAuthority231651.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232083

namespace LeftBound232087
def owner : Owner := ⟨.program ⟨257⟩, ⟨66540⟩⟩
def transferEvent : Nat := 232087
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232085 .coefficient, .predecessor 1 232086 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232085 .coefficient)
      LeftBound232083.bound (LeftBound232083.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232084RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232083.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232083.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232086 .coefficient)
      LeftAuthority231628.bound (LeftAuthority231628.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events904.exact231629RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority231628.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority231628.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232083.bound, LeftAuthority231628.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232083.bound, LeftAuthority231628.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232083.actual selector witness, LeftAuthority231628.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232087

namespace LeftBound232090
def owner : Owner := ⟨.program ⟨257⟩, ⟨66541⟩⟩
def transferEvent : Nat := 232090
def frameStart : Nat := 231586
def rule : BoundRule := .identity (.predecessor 0 232089 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232089 .coefficient)
      LeftBound232087.bound (LeftBound232087.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232088RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232087.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232087.derived selector witness)

def rawBound : CoeffClass := LeftBound232087.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232087.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound232087.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound232090

namespace LeftBound232107
def owner : Owner := ⟨.program ⟨257⟩, ⟨69083⟩⟩
def transferEvent : Nat := 232107
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232105 .coefficient, .predecessor 1 232106 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232105 .coefficient)
      LeftBound232090.bound (LeftBound232090.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound232090.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232106 .coefficient)
      LeftAuthority232103.bound (LeftAuthority232103.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority232103.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound232090.bound, LeftAuthority232103.bound]
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232090.bound, LeftAuthority232103.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound232090.actual selector witness, LeftAuthority232103.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232107

namespace LeftBound232110
def owner : Owner := ⟨.program ⟨257⟩, ⟨69084⟩⟩
def transferEvent : Nat := 232110
def frameStart : Nat := 231586
def rule : BoundRule := .identity (.predecessor 0 232109 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232109 .coefficient)
      LeftBound232107.bound (LeftBound232107.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound232107.derived selector witness)

def rawBound : CoeffClass := LeftBound232107.bound
def bound : CoeffClass := .finite ⟨1059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound232107.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound232107.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound232110

namespace LeftBound232116
def owner : Owner := ⟨.program ⟨257⟩, ⟨69085⟩⟩
def transferEvent : Nat := 232116
def frameStart : Nat := 231586
def rule : BoundRule := .product (.predecessor 0 232114 .coefficient) (.predecessor 1 232115 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232114 .coefficient)
      LeftAuthority232112.bound (LeftAuthority232112.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232113RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232112.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232112.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232115 .coefficient)
      LeftBound232110.bound (LeftBound232110.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232111RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound232110.bound, RecordedBoundRefines] <;> decide)
      (LeftBound232110.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority232112.bound LeftBound232110.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority232112.bound, LeftBound232110.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority232112.actual selector witness) * (LeftBound232110.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound232116

namespace LeftBound232192
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 232192
def frameStart : Nat := 231586
def rule : BoundRule := .sum [.predecessor 0 232190 .coefficient, .predecessor 1 232191 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 232190 .coefficient)
      LeftAuthority232188.bound (LeftAuthority232188.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232189RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232188.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232188.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 232191 .coefficient)
      LeftAuthority232185.bound (LeftAuthority232185.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events906.exact232186RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority232185.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority232185.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority232188.bound, LeftAuthority232185.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority232188.bound, LeftAuthority232185.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority232188.actual selector witness, LeftAuthority232185.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound232192

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
