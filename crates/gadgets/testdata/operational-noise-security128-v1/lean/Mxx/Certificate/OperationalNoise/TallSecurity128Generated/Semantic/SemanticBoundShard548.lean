import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard545
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard546
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard547

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound85970
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 85970
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85968 .coefficient, .predecessor 1 85969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85968 .coefficient)
      LeftBound85966.bound (LeftBound85966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85969 .coefficient)
      LeftAuthority85914.bound (LeftAuthority85914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85966.bound, LeftAuthority85914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85966.bound, LeftAuthority85914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85966.actual selector witness, LeftAuthority85914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85970

namespace LeftBound85974
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 85974
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85972 .coefficient, .predecessor 1 85973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85972 .coefficient)
      LeftBound85970.bound (LeftBound85970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85973 .coefficient)
      LeftAuthority85911.bound (LeftAuthority85911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85970.bound, LeftAuthority85911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85970.bound, LeftAuthority85911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85970.actual selector witness, LeftAuthority85911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85974

namespace LeftBound85978
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 85978
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85976 .coefficient, .predecessor 1 85977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85976 .coefficient)
      LeftBound85974.bound (LeftBound85974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85977 .coefficient)
      LeftAuthority85908.bound (LeftAuthority85908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85974.bound, LeftAuthority85908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85974.bound, LeftAuthority85908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85974.actual selector witness, LeftAuthority85908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85978

namespace LeftBound85982
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 85982
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85980 .coefficient, .predecessor 1 85981 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85980 .coefficient)
      LeftBound85978.bound (LeftBound85978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85981 .coefficient)
      LeftAuthority85905.bound (LeftAuthority85905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85978.bound, LeftAuthority85905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85978.bound, LeftAuthority85905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85978.actual selector witness, LeftAuthority85905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85982

namespace LeftBound85986
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 85986
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85984 .coefficient, .predecessor 1 85985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85984 .coefficient)
      LeftBound85982.bound (LeftBound85982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85985 .coefficient)
      LeftAuthority85902.bound (LeftAuthority85902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85982.bound, LeftAuthority85902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85982.bound, LeftAuthority85902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85982.actual selector witness, LeftAuthority85902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85986

namespace LeftBound85990
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 85990
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85988 .coefficient, .predecessor 1 85989 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85988 .coefficient)
      LeftBound85986.bound (LeftBound85986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85989 .coefficient)
      LeftAuthority85899.bound (LeftAuthority85899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85986.bound, LeftAuthority85899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85986.bound, LeftAuthority85899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85986.actual selector witness, LeftAuthority85899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85990

namespace LeftBound85994
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 85994
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85992 .coefficient, .predecessor 1 85993 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85992 .coefficient)
      LeftBound85990.bound (LeftBound85990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85993 .coefficient)
      LeftAuthority85896.bound (LeftAuthority85896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85990.bound, LeftAuthority85896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85990.bound, LeftAuthority85896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85990.actual selector witness, LeftAuthority85896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85994

namespace LeftBound85998
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 85998
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 85996 .coefficient, .predecessor 1 85997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 85996 .coefficient)
      LeftBound85994.bound (LeftBound85994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 85997 .coefficient)
      LeftAuthority85893.bound (LeftAuthority85893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85994.bound, LeftAuthority85893.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85994.bound, LeftAuthority85893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85994.actual selector witness, LeftAuthority85893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound85998

namespace LeftBound86002
def owner : Owner := ⟨.program ⟨257⟩, ⟨7324⟩⟩
def transferEvent : Nat := 86002
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 86000 .coefficient, .predecessor 1 86001 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86000 .coefficient)
      LeftBound85998.bound (LeftBound85998.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85999RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85998.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85998.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86001 .coefficient)
      LeftAuthority85890.bound (LeftAuthority85890.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85891RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85890.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85890.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound85998.bound, LeftAuthority85890.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound85998.bound, LeftAuthority85890.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound85998.actual selector witness, LeftAuthority85890.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86002

namespace LeftBound86006
def owner : Owner := ⟨.program ⟨257⟩, ⟨7325⟩⟩
def transferEvent : Nat := 86006
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 86004 .coefficient, .predecessor 1 86005 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86004 .coefficient)
      LeftBound86002.bound (LeftBound86002.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact86003RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86002.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86002.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86005 .coefficient)
      LeftAuthority85887.bound (LeftAuthority85887.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85888RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85887.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85887.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86002.bound, LeftAuthority85887.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86002.bound, LeftAuthority85887.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound86002.actual selector witness, LeftAuthority85887.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86006

namespace LeftBound86010
def owner : Owner := ⟨.program ⟨257⟩, ⟨69114⟩⟩
def transferEvent : Nat := 86010
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 86008 .coefficient, .predecessor 1 86009 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86008 .coefficient)
      LeftBound86006.bound (LeftBound86006.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact86007RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86006.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86006.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86009 .coefficient)
      LeftBound85866.bound (LeftBound85866.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85885RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound85866.bound, RecordedBoundRefines] <;> decide)
      (LeftBound85866.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86006.bound, LeftBound85866.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86006.bound, LeftBound85866.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound86006.actual selector witness, LeftBound85866.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86010

namespace LeftBound86014
def owner : Owner := ⟨.program ⟨257⟩, ⟨71438⟩⟩
def transferEvent : Nat := 86014
def frameStart : Nat := 85336
def rule : BoundRule := .product (.predecessor 0 86012 .coefficient) (.predecessor 1 86013 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86012 .coefficient)
      LeftBound86010.bound (LeftBound86010.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact86011RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86010.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86010.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86013 .coefficient)
      LeftAuthority85851.bound (LeftAuthority85851.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85852RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85851.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85851.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound86010.bound LeftAuthority85851.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86010.bound, LeftAuthority85851.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound86010.actual selector witness) * (LeftAuthority85851.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86014

namespace LeftBound86093
def owner : Owner := ⟨.program ⟨257⟩, ⟨67588⟩⟩
def transferEvent : Nat := 86093
def frameStart : Nat := 85336
def rule : BoundRule := .product (.predecessor 0 86091 .coefficient) (.predecessor 1 86092 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86091 .coefficient)
      LeftAuthority85862.bound (LeftAuthority85862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events335.exact85863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority85862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority85862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86092 .coefficient)
      LeftAuthority86089.bound (LeftAuthority86089.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86090RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86089.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86089.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority85862.bound LeftAuthority86089.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority85862.bound, LeftAuthority86089.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority85862.actual selector witness) * (LeftAuthority86089.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound86093

namespace LeftBound86101
def owner : Owner := ⟨.program ⟨257⟩, ⟨67593⟩⟩
def transferEvent : Nat := 86101
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 86099 .coefficient, .predecessor 1 86100 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86099 .coefficient)
      LeftAuthority86097.bound (LeftAuthority86097.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86098RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority86097.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority86097.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86100 .coefficient)
      LeftBound86093.bound (LeftBound86093.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86095RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86093.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86093.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority86097.bound, LeftBound86093.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority86097.bound, LeftBound86093.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority86097.actual selector witness, LeftBound86093.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86101

namespace LeftBound86105
def owner : Owner := ⟨.program ⟨257⟩, ⟨71442⟩⟩
def transferEvent : Nat := 86105
def frameStart : Nat := 85336
def rule : BoundRule := .sum [.predecessor 0 86103 .coefficient, .predecessor 1 86104 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86103 .coefficient)
      LeftBound86101.bound (LeftBound86101.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86102RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86101.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86101.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86104 .coefficient)
      LeftBound86014.bound (LeftBound86014.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86087RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound86014.bound, RecordedBoundRefines] <;> decide)
      (LeftBound86014.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound86101.bound, LeftBound86014.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound86101.bound, LeftBound86014.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound86101.actual selector witness, LeftBound86014.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86105

namespace LeftBound86152
def owner : Owner := ⟨.program ⟨257⟩, ⟨71440⟩⟩
def transferEvent : Nat := 86152
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 86150 .coefficient, .predecessor 1 86151 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 86150 .coefficient)
      LeftBound84743.bound (LeftBound84743.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events336.exact86149RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84743.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84743.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 86151 .coefficient)
      LeftBound84658.bound (LeftBound84658.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events330.exact84733RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound84658.bound, RecordedBoundRefines] <;> decide)
      (LeftBound84658.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound84743.bound, LeftBound84658.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound84743.bound, LeftBound84658.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound84743.actual selector witness, LeftBound84658.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound86152

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
