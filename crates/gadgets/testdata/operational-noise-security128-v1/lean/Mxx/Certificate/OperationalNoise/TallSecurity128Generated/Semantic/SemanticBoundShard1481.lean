import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticAuthority
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard053
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1448
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic.SemanticBoundShard1480

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftBound220917
def owner : Owner := ⟨.program ⟨257⟩, ⟨31829⟩⟩
def transferEvent : Nat := 220917
def frameStart : Nat := 220878
def rule : BoundRule := .identity (.predecessor 0 220916 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220916 .coefficient)
      LeftAuthority220914.bound (LeftAuthority220914.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events862.exact220915RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220914.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220914.derived selector witness)

def rawBound : CoeffClass := LeftAuthority220914.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220914.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftAuthority220914.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound220917

namespace LeftBound220934
def owner : Owner := ⟨.program ⟨257⟩, ⟨33306⟩⟩
def transferEvent : Nat := 220934
def frameStart : Nat := 220878
def rule : BoundRule := .sum [.predecessor 0 220932 .coefficient, .predecessor 1 220933 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220932 .coefficient)
      LeftBound220917.bound (LeftBound220917.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound220917.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220933 .coefficient)
      LeftAuthority220930.bound (LeftAuthority220930.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftAuthority220930.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220917.bound, LeftAuthority220930.bound]
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220917.bound, LeftAuthority220930.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220917.actual selector witness, LeftAuthority220930.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220934

namespace LeftBound220937
def owner : Owner := ⟨.program ⟨257⟩, ⟨33307⟩⟩
def transferEvent : Nat := 220937
def frameStart : Nat := 220878
def rule : BoundRule := .identity (.predecessor 0 220936 .coefficient)
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220936 .coefficient)
      LeftBound220934.bound (LeftBound220934.actual selector witness) := by
  refine .predecessor (rawTerms := none) (by rfl) ?_
  exact .resultCoefficient (by decide) (by rfl) (LeftBound220934.derived selector witness)

def rawBound : CoeffClass := LeftBound220934.bound
def bound : CoeffClass := .finite ⟨6, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220934.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := LeftBound220934.actual selector witness
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .identity (by rfl) (input0 selector witness)
end LeftBound220937

namespace LeftBound220943
def owner : Owner := ⟨.program ⟨257⟩, ⟨33308⟩⟩
def transferEvent : Nat := 220943
def frameStart : Nat := 220878
def rule : BoundRule := .product (.predecessor 0 220941 .coefficient) (.predecessor 1 220942 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220941 .coefficient)
      LeftAuthority220939.bound (LeftAuthority220939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220942 .coefficient)
      LeftBound220937.bound (LeftBound220937.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220938RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220937.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220937.derived selector witness)

def rawBound : CoeffClass := productWithFactor 32768 LeftAuthority220939.bound LeftBound220937.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220939.bound, LeftBound220937.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 32768 * (LeftAuthority220939.actual selector witness) * (LeftBound220937.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 32768) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220943

namespace LeftBound220951
def owner : Owner := ⟨.program ⟨257⟩, ⟨33309⟩⟩
def transferEvent : Nat := 220951
def frameStart : Nat := 220878
def rule : BoundRule := .sum [.predecessor 0 220949 .coefficient, .predecessor 1 220950 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220949 .coefficient)
      LeftAuthority220947.bound (LeftAuthority220947.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220948RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220947.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220947.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220950 .coefficient)
      LeftBound220943.bound (LeftBound220943.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220945RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220943.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220943.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority220947.bound, LeftBound220943.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220947.bound, LeftBound220943.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority220947.actual selector witness, LeftBound220943.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220951

namespace LeftBound220955
def owner : Owner := ⟨.program ⟨257⟩, ⟨33886⟩⟩
def transferEvent : Nat := 220955
def frameStart : Nat := 220878
def rule : BoundRule := .product (.predecessor 0 220953 .coefficient) (.predecessor 1 220954 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220953 .coefficient)
      LeftBound220951.bound (LeftBound220951.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220952RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220951.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220951.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220954 .coefficient)
      LeftAuthority220928.bound (LeftAuthority220928.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220929RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220928.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220928.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound220951.bound LeftAuthority220928.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220951.bound, LeftAuthority220928.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound220951.actual selector witness) * (LeftAuthority220928.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220955

namespace LeftBound220966
def owner : Owner := ⟨.program ⟨257⟩, ⟨32104⟩⟩
def transferEvent : Nat := 220966
def frameStart : Nat := 220878
def rule : BoundRule := .product (.predecessor 0 220964 .coefficient) (.predecessor 1 220965 .coefficient) ⟨false, true, none, none, some 1⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220964 .coefficient)
      LeftAuthority220939.bound (LeftAuthority220939.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220940RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220939.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220939.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220965 .coefficient)
      LeftAuthority220962.bound (LeftAuthority220962.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220963RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220962.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220962.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1 LeftAuthority220939.bound LeftAuthority220962.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220939.bound, LeftAuthority220962.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1 * (LeftAuthority220939.actual selector witness) * (LeftAuthority220962.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 1) (rightColumns := 1) (ringDimension := 32768) (factor := 1) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220966

namespace LeftBound220974
def owner : Owner := ⟨.program ⟨257⟩, ⟨32105⟩⟩
def transferEvent : Nat := 220974
def frameStart : Nat := 220878
def rule : BoundRule := .sum [.predecessor 0 220972 .coefficient, .predecessor 1 220973 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220972 .coefficient)
      LeftAuthority220970.bound (LeftAuthority220970.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220971RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority220970.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority220970.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220973 .coefficient)
      LeftBound220966.bound (LeftBound220966.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220968RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220966.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220966.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftAuthority220970.bound, LeftBound220966.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority220970.bound, LeftBound220966.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftAuthority220970.actual selector witness, LeftBound220966.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220974

namespace LeftBound220978
def owner : Owner := ⟨.program ⟨257⟩, ⟨33891⟩⟩
def transferEvent : Nat := 220978
def frameStart : Nat := 220878
def rule : BoundRule := .sum [.predecessor 0 220976 .coefficient, .predecessor 1 220977 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220976 .coefficient)
      LeftBound220974.bound (LeftBound220974.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220975RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220974.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220974.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220977 .coefficient)
      LeftBound220955.bound (LeftBound220955.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220960RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220955.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220955.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220974.bound, LeftBound220955.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220974.bound, LeftBound220955.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220974.actual selector witness, LeftBound220955.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220978

namespace LeftBound220991
def owner : Owner := ⟨.program ⟨257⟩, ⟨33888⟩⟩
def transferEvent : Nat := 220991
def frameStart : Nat := 0
def rule : BoundRule := .sum [.predecessor 0 220989 .coefficient, .predecessor 1 220990 .coefficient]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220989 .coefficient)
      LeftBound220820.bound (LeftBound220820.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220988RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220820.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220820.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220990 .coefficient)
      LeftBound220803.bound (LeftBound220803.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events862.exact220810RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220803.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220803.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220820.bound, LeftBound220803.bound]
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220820.bound, LeftBound220803.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220820.actual selector witness, LeftBound220803.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220991

namespace LeftBound220994
def owner : Owner := ⟨.program ⟨257⟩, ⟨33888⟩⟩
def transferEvent : Nat := 220994
def frameStart : Nat := 0
def rule : BoundRule := .sum [.result 220988 .summary, .result 220810 .summary]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220988 .summary)
      LeftBound220822.bound (LeftBound220822.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨32695⟩⟩) (rawTerms := some (Proof.Events863.exact220988RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound220822.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220810 .summary)
      LeftBound220805.bound (LeftBound220805.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33887⟩⟩) (rawTerms := some (Proof.Events862.exact220810RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound220805.derived selector witness)

def rawBound : CoeffClass := addKnownList [LeftBound220822.bound, LeftBound220805.bound]
def bound : CoeffClass := .finite ⟨32189200113375081643992404983808, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220822.bound, LeftBound220805.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := [LeftBound220822.actual selector witness, LeftBound220805.actual selector witness].sum
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .sum (by rfl) (.cons (input0 selector witness) (.cons (input1 selector witness) (.nil)))
end LeftBound220994

namespace LeftBound220998
def owner : Owner := ⟨.program ⟨257⟩, ⟨33889⟩⟩
def transferEvent : Nat := 220998
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 220996 .coefficient) (.predecessor 1 220997 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 220996 .coefficient)
      LeftBound220991.bound (LeftBound220991.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact220995RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound220991.bound, RecordedBoundRefines] <;> decide)
      (LeftBound220991.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 220997 .coefficient)
      LeftBound15821.bound (LeftBound15821.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events061.exact15822RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound15821.bound, RecordedBoundRefines] <;> decide)
      (LeftBound15821.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound220991.bound LeftBound15821.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220991.bound, LeftBound15821.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound220991.actual selector witness) * (LeftBound15821.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound220998

namespace LeftBound220999
def owner : Owner := ⟨.program ⟨257⟩, ⟨33889⟩⟩
def transferEvent : Nat := 220999
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨7145⟩⟩]⟩ [⟨.result 15818 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 15818 .coefficient)
      LeftAuthority15817.bound (LeftAuthority15817.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨7145⟩⟩) (rawTerms := some (Proof.Events061.exact15818RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority15817.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority15817.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority15817.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority15817.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority15817.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound220999

namespace LeftBound221000
def owner : Owner := ⟨.program ⟨257⟩, ⟨33889⟩⟩
def transferEvent : Nat := 221000
def frameStart : Nat := 0
def rule : BoundRule := .product (.result 220995 .summary) (.transfer 220999) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 220995 .summary)
      LeftBound220994.bound (LeftBound220994.actual selector witness) := by
  refine .resultSummary (resultOwner := ⟨.program ⟨257⟩, ⟨33888⟩⟩) (rawTerms := some (Proof.Events863.exact220995RawTerms)) (by decide) ?_
  exact .resultExactSummary (by rfl) (LeftBound220994.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.transfer 220999)
      LeftBound220999.bound (LeftBound220999.actual selector witness) := by
  exact .transfer (LeftBound220999.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound220994.bound LeftBound220999.bound
def bound : CoeffClass := .finite ⟨345628904428363669605693235694606923857920, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound220994.bound, LeftBound220999.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound220994.actual selector witness) * (LeftBound220999.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound221000

namespace LeftBound221015
def owner : Owner := ⟨.program ⟨257⟩, ⟨23867⟩⟩
def transferEvent : Nat := 221015
def frameStart : Nat := 0
def rule : BoundRule := .product (.predecessor 0 221013 .coefficient) (.predecessor 1 221014 .coefficient) ⟨false, false, none, none, none⟩
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 0 221013 .coefficient)
      LeftBound215032.bound (LeftBound215032.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events839.exact215036RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftBound215032.bound, RecordedBoundRefines] <;> decide)
      (LeftBound215032.derived selector witness)

theorem input1 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.predecessor 1 221014 .coefficient)
      LeftAuthority221011.bound (LeftAuthority221011.actual selector witness) := by
  refine .predecessor (rawTerms := some (Proof.Events863.exact221012RawTerms)) (by rfl) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority221011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority221011.derived selector witness)

def rawBound : CoeffClass := productWithFactor 1310720 LeftBound215032.bound LeftAuthority221011.bound
def bound : CoeffClass := .large
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftBound215032.bound, LeftAuthority221011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := 1310720 * (LeftBound215032.actual selector witness) * (LeftAuthority221011.actual selector witness)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .product (leftRows := 1) (leftColumns := 40) (rightRows := 40) (rightColumns := 40) (ringDimension := 32768) (factor := 1310720) (by rfl) (by decide) (input0 selector witness) (input1 selector witness)
end LeftBound221015

namespace LeftBound221016
def owner : Owner := ⟨.program ⟨257⟩, ⟨23867⟩⟩
def transferEvent : Nat := 221016
def frameStart : Nat := 0
def rule : BoundRule := .monomialProduct ⟨[], [⟨.program ⟨257⟩, ⟨23865⟩⟩]⟩ [⟨.result 221012 .coefficient, false, none⟩]
theorem input0 (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundInputAt history owner (.result 221012 .coefficient)
      LeftAuthority221011.bound (LeftAuthority221011.actual selector witness) := by
  refine .result (resultOwner := ⟨.program ⟨257⟩, ⟨23865⟩⟩) (rawTerms := some (Proof.Events863.exact221012RawTerms)) (by decide) ?_
  exact .resultExactCoefficient (by rfl)
      (by dsimp [LeftAuthority221011.bound, RecordedBoundRefines] <;> decide)
      (LeftAuthority221011.derived selector witness)

def rawBound : CoeffClass := productNonempty LeftAuthority221011.bound []
def bound : CoeffClass := .finite ⟨8192, by decide⟩
theorem rawBound_eq_bound : rawBound = bound := by
  dsimp [rawBound, bound, LeftAuthority221011.bound, addKnownList, addKnown, productWithFactor,
    scaleMagnitude, scaleValue, productNonempty] <;> decide
def actual (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) : Nat := (LeftAuthority221011.actual selector witness) * ([].prod)
theorem derived (selector : Nat)
    (witness : Witness document history (some selector) 2596118388884299782786442882919966202757356512598865539416966488295494748974456019701577684907109706420296806278632487222012309918168506832951461416943272286629018402817) :
    BoundDerivedAt history transferEvent frameStart owner rule bound
      (actual selector witness) := by
  rw [← rawBound_eq_bound]
  unfold rule rawBound actual
  refine .monomialProduct (by rfl) (.intro (input0 selector witness)) (.nil)
end LeftBound221016

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
