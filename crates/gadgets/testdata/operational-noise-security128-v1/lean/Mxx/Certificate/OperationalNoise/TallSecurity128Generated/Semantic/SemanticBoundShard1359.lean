import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1358

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound202866
def owner : Owner := ⟨.program ⟨257⟩, ⟨69097⟩⟩
def transferEvent : Nat := 202866
def frameStart : Nat := 202336
def rule : BoundRule := .product (.predecessor 0 202864 .coefficient) (.predecessor 1 202865 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202864 .coefficient)
      LeftAuthority202862.bound (LeftAuthority202862.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202863RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202862.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202862.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202865 .coefficient)
      LeftBound202860.bound (LeftBound202860.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202861RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202860.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202860.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority202862.bound LeftBound202860.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority202862.bound, LeftBound202860.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority202862.actual selector witness) * (LeftBound202860.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound202866

namespace LeftBound202942
def owner : Owner := ⟨.program ⟨257⟩, ⟨7309⟩⟩
def transferEvent : Nat := 202942
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202940 .coefficient, .predecessor 1 202941 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202940 .coefficient)
      LeftAuthority202938.bound (LeftAuthority202938.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202939RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202938.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202938.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202941 .coefficient)
      LeftAuthority202935.bound (LeftAuthority202935.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202936RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202935.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202935.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority202938.bound, LeftAuthority202935.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority202938.bound, LeftAuthority202935.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority202938.actual selector witness, LeftAuthority202935.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202942

namespace LeftBound202946
def owner : Owner := ⟨.program ⟨257⟩, ⟨7310⟩⟩
def transferEvent : Nat := 202946
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202944 .coefficient, .predecessor 1 202945 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202944 .coefficient)
      LeftBound202942.bound (LeftBound202942.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202943RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202942.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202942.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202945 .coefficient)
      LeftAuthority202932.bound (LeftAuthority202932.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202933RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202932.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202932.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202942.bound, LeftAuthority202932.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202942.bound, LeftAuthority202932.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202942.actual selector witness, LeftAuthority202932.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202946

namespace LeftBound202950
def owner : Owner := ⟨.program ⟨257⟩, ⟨7311⟩⟩
def transferEvent : Nat := 202950
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202948 .coefficient, .predecessor 1 202949 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202948 .coefficient)
      LeftBound202946.bound (LeftBound202946.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202947RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202946.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202946.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202949 .coefficient)
      LeftAuthority202929.bound (LeftAuthority202929.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202930RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202929.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202929.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202946.bound, LeftAuthority202929.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202946.bound, LeftAuthority202929.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202946.actual selector witness, LeftAuthority202929.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202950

namespace LeftBound202954
def owner : Owner := ⟨.program ⟨257⟩, ⟨7312⟩⟩
def transferEvent : Nat := 202954
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202952 .coefficient, .predecessor 1 202953 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202952 .coefficient)
      LeftBound202950.bound (LeftBound202950.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202951RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202950.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202950.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202953 .coefficient)
      LeftAuthority202926.bound (LeftAuthority202926.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202927RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202926.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202926.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202950.bound, LeftAuthority202926.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202950.bound, LeftAuthority202926.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202950.actual selector witness, LeftAuthority202926.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202954

namespace LeftBound202958
def owner : Owner := ⟨.program ⟨257⟩, ⟨7313⟩⟩
def transferEvent : Nat := 202958
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202956 .coefficient, .predecessor 1 202957 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202956 .coefficient)
      LeftBound202954.bound (LeftBound202954.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202955RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202954.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202954.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202957 .coefficient)
      LeftAuthority202923.bound (LeftAuthority202923.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202924RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202923.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202923.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202954.bound, LeftAuthority202923.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202954.bound, LeftAuthority202923.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202954.actual selector witness, LeftAuthority202923.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202958

namespace LeftBound202962
def owner : Owner := ⟨.program ⟨257⟩, ⟨7314⟩⟩
def transferEvent : Nat := 202962
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202960 .coefficient, .predecessor 1 202961 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202960 .coefficient)
      LeftBound202958.bound (LeftBound202958.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202959RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202958.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202958.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202961 .coefficient)
      LeftAuthority202920.bound (LeftAuthority202920.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202921RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202920.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202920.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202958.bound, LeftAuthority202920.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202958.bound, LeftAuthority202920.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202958.actual selector witness, LeftAuthority202920.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202962

namespace LeftBound202966
def owner : Owner := ⟨.program ⟨257⟩, ⟨7315⟩⟩
def transferEvent : Nat := 202966
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202964 .coefficient, .predecessor 1 202965 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202964 .coefficient)
      LeftBound202962.bound (LeftBound202962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202962.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202962.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202965 .coefficient)
      LeftAuthority202917.bound (LeftAuthority202917.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202918RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202917.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202917.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202962.bound, LeftAuthority202917.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202962.bound, LeftAuthority202917.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202962.actual selector witness, LeftAuthority202917.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202966

namespace LeftBound202970
def owner : Owner := ⟨.program ⟨257⟩, ⟨7316⟩⟩
def transferEvent : Nat := 202970
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202968 .coefficient, .predecessor 1 202969 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202968 .coefficient)
      LeftBound202966.bound (LeftBound202966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202967RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202966.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202969 .coefficient)
      LeftAuthority202914.bound (LeftAuthority202914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202914.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202966.bound, LeftAuthority202914.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202966.bound, LeftAuthority202914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202966.actual selector witness, LeftAuthority202914.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202970

namespace LeftBound202974
def owner : Owner := ⟨.program ⟨257⟩, ⟨7317⟩⟩
def transferEvent : Nat := 202974
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202972 .coefficient, .predecessor 1 202973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202972 .coefficient)
      LeftBound202970.bound (LeftBound202970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202970.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202973 .coefficient)
      LeftAuthority202911.bound (LeftAuthority202911.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202912RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202911.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202911.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202970.bound, LeftAuthority202911.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202970.bound, LeftAuthority202911.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202970.actual selector witness, LeftAuthority202911.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202974

namespace LeftBound202978
def owner : Owner := ⟨.program ⟨257⟩, ⟨7318⟩⟩
def transferEvent : Nat := 202978
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202976 .coefficient, .predecessor 1 202977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202976 .coefficient)
      LeftBound202974.bound (LeftBound202974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202977 .coefficient)
      LeftAuthority202908.bound (LeftAuthority202908.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202909RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202908.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202908.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202974.bound, LeftAuthority202908.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202974.bound, LeftAuthority202908.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202974.actual selector witness, LeftAuthority202908.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202978

namespace LeftBound202982
def owner : Owner := ⟨.program ⟨257⟩, ⟨7319⟩⟩
def transferEvent : Nat := 202982
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202980 .coefficient, .predecessor 1 202981 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202980 .coefficient)
      LeftBound202978.bound (LeftBound202978.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202979RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202978.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202978.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202981 .coefficient)
      LeftAuthority202905.bound (LeftAuthority202905.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202906RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202905.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202905.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202978.bound, LeftAuthority202905.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202978.bound, LeftAuthority202905.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202978.actual selector witness, LeftAuthority202905.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202982

namespace LeftBound202986
def owner : Owner := ⟨.program ⟨257⟩, ⟨7320⟩⟩
def transferEvent : Nat := 202986
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202984 .coefficient, .predecessor 1 202985 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202984 .coefficient)
      LeftBound202982.bound (LeftBound202982.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202983RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202982.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202982.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202985 .coefficient)
      LeftAuthority202902.bound (LeftAuthority202902.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202903RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202902.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202902.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202982.bound, LeftAuthority202902.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202982.bound, LeftAuthority202902.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202982.actual selector witness, LeftAuthority202902.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202986

namespace LeftBound202990
def owner : Owner := ⟨.program ⟨257⟩, ⟨7321⟩⟩
def transferEvent : Nat := 202990
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202988 .coefficient, .predecessor 1 202989 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202988 .coefficient)
      LeftBound202986.bound (LeftBound202986.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202987RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202986.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202986.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202989 .coefficient)
      LeftAuthority202899.bound (LeftAuthority202899.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202900RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202899.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202899.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202986.bound, LeftAuthority202899.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202986.bound, LeftAuthority202899.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202986.actual selector witness, LeftAuthority202899.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202990

namespace LeftBound202994
def owner : Owner := ⟨.program ⟨257⟩, ⟨7322⟩⟩
def transferEvent : Nat := 202994
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202992 .coefficient, .predecessor 1 202993 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202992 .coefficient)
      LeftBound202990.bound (LeftBound202990.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202991RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202990.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202990.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202993 .coefficient)
      LeftAuthority202896.bound (LeftAuthority202896.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202897RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202896.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202896.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202990.bound, LeftAuthority202896.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202990.bound, LeftAuthority202896.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202990.actual selector witness, LeftAuthority202896.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202994

namespace LeftBound202998
def owner : Owner := ⟨.program ⟨257⟩, ⟨7323⟩⟩
def transferEvent : Nat := 202998
def frameStart : Nat := 202336
def rule : BoundRule := .sum [.predecessor 0 202996 .coefficient, .predecessor 1 202997 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 202996 .coefficient)
      LeftBound202994.bound (LeftBound202994.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound202994.bound, RecordedBoundRefines] <;> decide)
      (LeftBound202994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 202997 .coefficient)
      LeftAuthority202893.bound (LeftAuthority202893.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events792.exact202894RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority202893.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority202893.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound202994.bound, LeftAuthority202893.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound202994.bound, LeftAuthority202893.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound202994.actual selector witness, LeftAuthority202893.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound202998

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
