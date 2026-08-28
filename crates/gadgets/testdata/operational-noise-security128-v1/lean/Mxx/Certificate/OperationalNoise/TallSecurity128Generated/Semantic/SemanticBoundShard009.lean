import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard007
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard008

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound2995
def owner : Owner := ⟨.program ⟨257⟩, ⟨54280⟩⟩
def transferEvent : Nat := 2995
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2993 .coefficient, .predecessor 1 2994 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 2993 .coefficient)
      LeftBound2991.bound (LeftBound2991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2992RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 2994 .coefficient)
      LeftBound2930.bound (LeftBound2930.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2932RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2930.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2991.bound, LeftBound2930.bound]
def bound : CoeffClass := .finite ⟨1150828286136974432938179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2991.bound, LeftBound2930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound2991.actual selector witness, LeftBound2930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2995

namespace LeftBound2999
def owner : Owner := ⟨.program ⟨257⟩, ⟨57260⟩⟩
def transferEvent : Nat := 2999
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 2997 .coefficient, .predecessor 1 2998 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 2997 .coefficient)
      LeftBound2995.bound (LeftBound2995.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2996RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2995.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2995.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 2998 .coefficient)
      LeftBound2922.bound (LeftBound2922.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2922.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2922.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2995.bound, LeftBound2922.bound]
def bound : CoeffClass := .finite ⟨1371606415754681672436099, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2995.bound, LeftBound2922.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound2995.actual selector witness, LeftBound2922.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound2999

namespace LeftBound3003
def owner : Owner := ⟨.program ⟨257⟩, ⟨60240⟩⟩
def transferEvent : Nat := 3003
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3001 .coefficient, .predecessor 1 3002 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3001 .coefficient)
      LeftBound2999.bound (LeftBound2999.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3000RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2999.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2999.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3002 .coefficient)
      LeftBound2914.bound (LeftBound2914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2916RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2914.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound2999.bound, LeftBound2914.bound]
def bound : CoeffClass := .finite ⟨1593837033067242249035979, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound2999.bound, LeftBound2914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound2999.actual selector witness, LeftBound2914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3003

namespace LeftBound3007
def owner : Owner := ⟨.program ⟨257⟩, ⟨63220⟩⟩
def transferEvent : Nat := 3007
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3005 .coefficient, .predecessor 1 3006 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3005 .coefficient)
      LeftBound3003.bound (LeftBound3003.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3004RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3003.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3003.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3006 .coefficient)
      LeftBound2906.bound (LeftBound2906.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2908RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2906.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2906.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3003.bound, LeftBound2906.bound]
def bound : CoeffClass := .finite ⟨1818214806102629497873539, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3003.bound, LeftBound2906.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3003.actual selector witness, LeftBound2906.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3007

namespace LeftBound3011
def owner : Owner := ⟨.program ⟨257⟩, ⟨67080⟩⟩
def transferEvent : Nat := 3011
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3009 .coefficient, .predecessor 1 3010 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3009 .coefficient)
      LeftBound3007.bound (LeftBound3007.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3008RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3007.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3007.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3010 .coefficient)
      LeftBound2898.bound (LeftBound2898.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2898.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2898.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3007.bound, LeftBound2898.bound]
def bound : CoeffClass := .finite ⟨2044702714934587786668819, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3007.bound, LeftBound2898.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3007.actual selector witness, LeftBound2898.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3011

namespace LeftBound3015
def owner : Owner := ⟨.program ⟨257⟩, ⟨67081⟩⟩
def transferEvent : Nat := 3015
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3013 .coefficient, .predecessor 1 3014 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3013 .coefficient)
      LeftBound3011.bound (LeftBound3011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3011.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3011.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3014 .coefficient)
      LeftBound2890.bound (LeftBound2890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2892RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2890.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3011.bound, LeftBound2890.bound]
def bound : CoeffClass := .finite ⟨2271712485307633536959019, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3011.bound, LeftBound2890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3011.actual selector witness, LeftBound2890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3015

namespace LeftBound3019
def owner : Owner := ⟨.program ⟨257⟩, ⟨67082⟩⟩
def transferEvent : Nat := 3019
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3017 .coefficient, .predecessor 1 3018 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3017 .coefficient)
      LeftBound3015.bound (LeftBound3015.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3016RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3015.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3015.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3018 .coefficient)
      LeftBound2882.bound (LeftBound2882.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2884RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2882.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2882.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3015.bound, LeftBound2882.bound]
def bound : CoeffClass := .finite ⟨2499949335520533588602139, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3015.bound, LeftBound2882.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3015.actual selector witness, LeftBound2882.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3019

namespace LeftBound3023
def owner : Owner := ⟨.program ⟨257⟩, ⟨67083⟩⟩
def transferEvent : Nat := 3023
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3021 .coefficient, .predecessor 1 3022 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3021 .coefficient)
      LeftBound3019.bound (LeftBound3019.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3020RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3019.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3019.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3022 .coefficient)
      LeftBound2874.bound (LeftBound2874.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2876RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2874.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2874.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3019.bound, LeftBound2874.bound]
def bound : CoeffClass := .finite ⟨2728804713782791092959739, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3019.bound, LeftBound2874.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3019.actual selector witness, LeftBound2874.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3023

namespace LeftBound3027
def owner : Owner := ⟨.program ⟨257⟩, ⟨67084⟩⟩
def transferEvent : Nat := 3027
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3025 .coefficient, .predecessor 1 3026 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3025 .coefficient)
      LeftBound3023.bound (LeftBound3023.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3024RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3023.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3023.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3026 .coefficient)
      LeftBound2866.bound (LeftBound2866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2868RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3023.bound, LeftBound2866.bound]
def bound : CoeffClass := .finite ⟨2957926202950004710694499, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3023.bound, LeftBound2866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3023.actual selector witness, LeftBound2866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3027

namespace LeftBound3031
def owner : Owner := ⟨.program ⟨257⟩, ⟨67085⟩⟩
def transferEvent : Nat := 3031
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3029 .coefficient, .predecessor 1 3030 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3029 .coefficient)
      LeftBound3027.bound (LeftBound3027.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3028RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3027.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3027.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3030 .coefficient)
      LeftBound2858.bound (LeftBound2858.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2860RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2858.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2858.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3027.bound, LeftBound2858.bound]
def bound : CoeffClass := .finite ⟨3187511970717354526236219, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3027.bound, LeftBound2858.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3027.actual selector witness, LeftBound2858.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3031

namespace LeftBound3035
def owner : Owner := ⟨.program ⟨257⟩, ⟨67086⟩⟩
def transferEvent : Nat := 3035
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3033 .coefficient, .predecessor 1 3034 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3033 .coefficient)
      LeftBound3031.bound (LeftBound3031.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3032RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3031.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3031.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3034 .coefficient)
      LeftBound2850.bound (LeftBound2850.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2850.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2850.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3031.bound, LeftBound2850.bound]
def bound : CoeffClass := .finite ⟨3417662756781096507033579, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3031.bound, LeftBound2850.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3031.actual selector witness, LeftBound2850.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3035

namespace LeftBound3039
def owner : Owner := ⟨.program ⟨257⟩, ⟨67087⟩⟩
def transferEvent : Nat := 3039
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3037 .coefficient, .predecessor 1 3038 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3037 .coefficient)
      LeftBound3035.bound (LeftBound3035.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3035.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3035.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3038 .coefficient)
      LeftBound2842.bound (LeftBound2842.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2844RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2842.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2842.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3035.bound, LeftBound2842.bound]
def bound : CoeffClass := .finite ⟨3648263642165693263543059, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3035.bound, LeftBound2842.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3035.actual selector witness, LeftBound2842.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3039

namespace LeftBound3043
def owner : Owner := ⟨.program ⟨257⟩, ⟨67088⟩⟩
def transferEvent : Nat := 3043
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3041 .coefficient, .predecessor 1 3042 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3041 .coefficient)
      LeftBound3039.bound (LeftBound3039.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3040RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3039.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3039.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3042 .coefficient)
      LeftBound2834.bound (LeftBound2834.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2836RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2834.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2834.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3039.bound, LeftBound2834.bound]
def bound : CoeffClass := .finite ⟨3878994884184198780231459, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3039.bound, LeftBound2834.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3039.actual selector witness, LeftBound2834.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3043

namespace LeftBound3047
def owner : Owner := ⟨.program ⟨257⟩, ⟨67609⟩⟩
def transferEvent : Nat := 3047
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 3045 .coefficient, .predecessor 1 3046 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3045 .coefficient)
      LeftBound3043.bound (LeftBound3043.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3044RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3043.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3043.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3046 .coefficient)
      LeftBound2826.bound (LeftBound2826.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact2828RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound2826.bound, RecordedBoundRefines] <;> decide)
      (LeftBound2826.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound3043.bound, LeftBound2826.bound]
def bound : CoeffClass := .finite ⟨8101376613122849735629179, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3043.bound, LeftBound2826.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound3043.actual selector witness, LeftBound2826.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound3047

namespace LeftBound3051
def owner : Owner := ⟨.program ⟨257⟩, ⟨67610⟩⟩
def transferEvent : Nat := 3051
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 3049 .coefficient) (.predecessor 1 3050 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3049 .coefficient)
      LeftBound3047.bound (LeftBound3047.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events011.exact3048RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound3047.bound, RecordedBoundRefines] <;> decide)
      (LeftBound3047.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3050 .coefficient)
      LeftAuthority2324.bound (LeftAuthority2324.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events009.exact2325RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority2324.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority2324.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftBound3047.bound LeftAuthority2324.bound
def bound : CoeffClass := .finite ⟨12779664183506592515459045800984236610315689098079499208139574138121340648205591080964678382018650142870845762146940804687506692660818735551867909260499956473046718986730217969403959728762168606720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound3047.bound, LeftAuthority2324.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftBound3047.actual selector witness) * (LeftAuthority2324.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound3051

namespace LeftBound3574
def owner : Owner := ⟨.program ⟨257⟩, ⟨67587⟩⟩
def transferEvent : Nat := 3574
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 3572 .coefficient) (.predecessor 1 3573 .coefficient) ⟨true, true, none, some 1, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 3572 .coefficient)
      LeftAuthority3570.bound (LeftAuthority3570.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events013.exact3571RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority3570.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority3570.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 3573 .coefficient)
      LeftAuthority35.bound (LeftAuthority35.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events000.exact36RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority35.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority35.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority3570.bound LeftAuthority35.bound
def bound : CoeffClass := .finite ⟨4222381728938650955397720, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority3570.bound, LeftAuthority35.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority3570.actual selector witness) * (LeftAuthority35.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 1) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound3574

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
